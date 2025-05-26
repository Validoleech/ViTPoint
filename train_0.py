import torchvision
import yaml
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
from utils.loss import focal_bce, compute_val_loss, l1_offset, soft_nms_penalty, dice_loss
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
                        num_workers=cfg["data"]["num_workers"], shuffle=False, pin_memory=True)

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
    L_DICE = 1.0
    L_BCE = 1.5
    EPOCHS = args.epochs or cfg['train']['epochs_stage0']

    for ep in range(EPOCHS):
        model.train()
        running = 0
        for img, heat, off in tqdm(loader, desc=f"E{ep}"):
            w_before = model.det_head.weight.clone()
            img  = img.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True)
            off = off.to(device, non_blocking=True)
            opt.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype):
                heat_pr, off_pr, _ = model(img)
                loss_det = L_DICE * dice_loss(heat_pr, heat) + \
                L_BCE * focal_bce(heat_pr, heat, gamma=G)
                loss_off = l1_offset(off_pr, off, heat)
                loss_nms = soft_nms_penalty(heat_pr)
                loss = loss_det + L_NMS*loss_nms + L_OFF*loss_off
                pos_fraction = (heat_pr > 0.5).float().mean()
                writer.add_scalar("debug/pos_px_frac", pos_fraction, global_step)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.eval()
            writer.add_scalar("train/loss_det",   loss_det.item(),  global_step)
            writer.add_scalar("train/loss_off",   loss_off.item(),  global_step)
            writer.add_scalar("train/loss_total", loss.item(),      global_step)
            global_step += 1
            running += loss.item()
            if global_step % 600 == 0:
                model.eval()
                with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype):
                    # PR - F1
                    v_img, v_heat, v_off = next(iter(val_loader))
                    v_img = v_img.to(device)
                    v_heat = v_heat.to(device)
                    v_heat_pr, _, _ = model(v_img)
                    # gt_ratio = (v_heat > 0.5).float().mean().item()
                    # print(f"GT pos-fraction = {gt_ratio:.4f}")
                    P_sum = R_sum = 0
                    for b in range(v_heat_pr.size(0)):
                        P, R = pr_single_batch(v_heat_pr[b:b+1],  # [1,1,h,w]
                                            v_heat[b:b+1])
                        P_sum += P;  R_sum += R
                    P = P_sum / v_heat_pr.size(0)
                    R = R_sum / v_heat_pr.size(0)
                    F1 = 2*P*R / (P+R+1e-6)
                    writer.add_scalar("val/P", P, global_step)
                    writer.add_scalar("val/R", R, global_step)
                    writer.add_scalar("val/F1", 2*P*R/(P+R+1e-6), global_step)
                    ## VIS
                    heat_gt = F.interpolate(
                        v_heat,    size=v_img.shape[-2:], mode="nearest")
                    heat_pred = F.interpolate(
                        v_heat_pr, size=v_img.shape[-2:], mode="nearest")
                    heat_gt = heat_gt.repeat(1, 3, 1, 1)        # 1→3 channels
                    heat_pred = heat_pred.repeat(1, 3, 1, 1)
                    vis = torch.cat([v_img.clamp(0, 1).cpu(),
                                    heat_gt.cpu(),
                                    heat_pred.cpu()], dim=0)
                    writer.add_image('val/qualitative', torchvision.utils.make_grid(
                        vis,nrow=v_img.size(0), normalize=True, scale_each=True), global_step)
                    viz = torchvision.utils.make_grid(heat_pr[:8])
                    writer.add_image('val/heatmap', viz, global_step)
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
                   f"runs/stage0/checkpoint_stage1-{time.strftime("%Y%m%d-%H%M%S")}-epoch{ep}.pth")
    torch.save(model.state_dict(), f"checkpoint_stage0-{time.strftime("%Y%m%d-%H%M%S")}.pth")
    print(f"Stage-0 finished — saved to checkpoint_stage0-{time.strftime("%Y%m%d-%H%M%S")}.pth")

