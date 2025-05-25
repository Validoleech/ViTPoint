import csv
import yaml
import os
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
from utils.loss import focal_bce, compute_val_loss, l1_offset, soft_nms_penalty
from utils.eval import evaluate_detector_pr, val_step, pr_single_batch

if __name__ == "__main__":
    
    writer = SummaryWriter(log_dir="runs/stage0")
    global_step = 0
    
    parser = ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    

    ps_cfg   = PersistentSynthConfig()
    ds_train = PersistentSyntheticShapes("train", ps_cfg)
    ds_val   = PersistentSyntheticShapes("val",   ps_cfg)

    loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"],
                        num_workers=cfg["data"]["num_workers"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"],
                        num_workers=cfg["data"]["num_workers"], shuffle=True, pin_memory=True)

    # Model / Optimiser
    model = SuperPointViT(HeadConfig(cfg["model"]["dim_descriptor"]),
                        freeze_backbone=True).to(device)
    if cfg["model"].get("freeze_backbone_stage0", True):
        model.backbone.eval()
    opt = torch.optim.AdamW(list(model.fuse_conv.parameters()) +
                            list(model.det_head.parameters()) +
                            list(model.offset_head.parameters()) +
                            list(model.desc_head.parameters()),
                            lr=float(cfg["train"]["lr_heads"]))
    
    scaler = GradScaler()
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optim'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep = ckpt['epoch'] + 1
    else:
        start_ep = 0

    A = None #0.75
    G = 2.5
    L_NMS = 0.05
    L_OFF = 2.0
    EPOCHS = args.epochs or cfg['train']['epochs_stage0']

    for ep in range(EPOCHS):
        model.train()
        running = 0
        for img, heat, off in tqdm(loader, desc=f"E{ep}"):
            w_before = model.det_head.weight.clone()
            img  = img.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True)
            assert heat.sum() > 0, "No positive pixels in the label!"
            off = off.to(device, non_blocking=True)
            opt.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype):
                heat_pr, off_pr, _ = model(img)
                assert heat_pr.shape == heat.shape, \
                    f"Shape mismatch   pred {heat_pr.shape}   gt {heat.shape}"
                with torch.no_grad():
                    pos = heat.sum()
                    neg = heat.numel() - pos
                    # # add epsilon to avoid 0/0
                    # w_pos = neg / (pos + 1e-6)
                    # w_neg = pos / (neg + 1e-6)
                    if 'ema_pos' not in globals():
                        ema_pos = pos
                        ema_neg = neg
                    else:
                        ema_pos = 0.9 * ema_pos + 0.1 * pos
                        ema_neg = 0.9 * ema_neg + 0.1 * neg
                    # w_pos = (neg + 1e-6) / (pos + 1e-6)
                    max_ratio = 5.0 # 5.0
                    w_pos = torch.clamp(neg / pos, max=max_ratio)
                    w_neg = 1.0
                    A = w_pos * heat + w_neg * (1 - heat)
                loss_det = focal_bce(heat_pr, heat, alpha=A, gamma=G)
                loss_off = l1_offset(off_pr, off, heat)
                loss_nms = soft_nms_penalty(heat_pr)
                loss = loss_det + L_NMS*loss_nms + L_OFF*loss_off
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            writer.add_scalar("train/loss_det",   loss_det.item(),  global_step)
            writer.add_scalar("train/loss_off",   loss_off.item(),  global_step)
            writer.add_scalar("train/loss_total", loss.item(),      global_step)
            writer.add_scalar("debug/w_pos", w_pos,          global_step)
            writer.add_scalar("debug/w_neg", w_neg,          global_step)
            global_step += 1
            running += loss.item()
            if global_step % 1000 == 0:
                model.eval()
                with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype):
                    v_img, v_heat, v_off = next(iter(val_loader))
                    v_img = v_img.to(device)
                    v_heat = v_heat.to(device)
                    v_heat_pr, _, _ = model(v_img)
                    P, R = pr_single_batch(v_heat_pr, v_heat)
                    writer.add_scalar("val/P", P, global_step)
                    writer.add_scalar("val/R", R, global_step)
                    writer.add_scalar("val/F1", 2*P*R/(P+R+1e-6), global_step)
            model.train()
        print(f"Epoch {ep} | Loss {running/len(loader):.4f}")
        metrics = evaluate_detector_pr(model, val_loader, device=device)
        vloss = compute_val_loss(model, val_loader, val_step, device)
        print(f"  Val P:{metrics['P']:.3f} R:{metrics['R']:.3f} "
            f"F1:{metrics['F1']:.3f} AP:{metrics['AP']:.3f}  "
            f"loss:{vloss:.4f}")
        row = [ep,
            running/len(loader),          # train_loss
            vloss,                        # val_loss
            metrics['P'], metrics['R'],
            metrics['F1'], metrics['AP']]
        torch.save(model.state_dict(),
                   f"runs/stage0/checkpoint_stage0-{time.strftime("%Y%m%d-%H%M%S")}-epoch{ep}.pth")
    torch.save(model.state_dict(), f"checkpoint_stage0-{time.strftime("%Y%m%d-%H%M%S")}.pth")
    print(f"Stage-0 finished — saved to checkpoint_stage0-{time.strftime("%Y%m%d-%H%M%S")}.pth")

