import csv
import yaml
import os
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from argparse import ArgumentParser

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
from utils.loss import focal_bce, compute_val_loss, l1_offset, soft_nms_penalty
from utils.eval import evaluate_detector_pr, val_step_stage01

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)

    cfg = yaml.safe_load(open("config.yaml"))
    out_csv = f"data/stage0_metrics-{time.strftime("%Y%m%d-%H%M%S")}.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "P", "R", "F1", "AP"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    

    ps_cfg   = PersistentSynthConfig()
    ds_train = PersistentSyntheticShapes("train", ps_cfg)
    ds_val   = PersistentSyntheticShapes("val",   ps_cfg)

    loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"],
                        num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"],
                        num_workers=cfg["data"]["num_workers"])
    
    # tot_pos = 0
    # for _, h in DataLoader(loader, batch_size=64):
    #     tot_pos += h.sum().item()
    # print("average positives / image", tot_pos/len(loader))


    # Model / Optimiser
    model = SuperPointViT(HeadConfig(cfg["model"]["dim_descriptor"]),
                        freeze_backbone=True).to(device)

    opt = torch.optim.AdamW(list(model.det_head.parameters()) +
                            list(model.desc_head.parameters()),
                            lr=float(cfg["train"]["lr_heads"]))
    
    # n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("trainable parameters:", n_trainable)     # should be > 0
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.shape)

    # pos_per_img = 0
    # for _, heat, _ in tqdm(loader):
    #     pos_per_img += heat.sum().item()
    # print("avg positive patches per image:",
    #     pos_per_img / len(ds_train))
    scaler = GradScaler()
    
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optim'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep = ckpt['epoch'] + 1
    else:
        start_ep = 0

    EPOCHS = args.epochs or cfg.get('epochs_stage0', 2)
    for ep in range(EPOCHS):
        model.train()
        running = 0
        for img, heat, off in tqdm(loader, desc=f"E{ep}"):
            img  = img.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True)
            off = off.to(device, non_blocking=True)
            opt.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype):
                heat_pr, off_pr, _ = model(img)
                assert heat_pr.shape == heat.shape, \
                    f"Shape mismatch   pred {heat_pr.shape}   gt {heat.shape}"
                loss_det = focal_bce(heat_pr, heat)
                loss_off = l1_offset(off_pr, off, heat)
                loss_nms = soft_nms_penalty(heat_pr)
                loss = loss_det + 0.2*loss_nms + 2.0*loss_off
            scaler.scale(loss).backward()
            print("grad mean det_head:", model.det_head.weight.grad.abs().mean())
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        print(f"Epoch {ep} | Loss {running/len(loader):.4f}")
        metrics = evaluate_detector_pr(model, val_loader, device=device)
        vloss = compute_val_loss(model, val_loader, val_step_stage01, device)
        print(f"  Val P:{metrics['P']:.3f} R:{metrics['R']:.3f} "
            f"F1:{metrics['F1']:.3f} AP:{metrics['AP']:.3f}  "
            f"loss:{vloss:.4f}")
        row = [ep,
            running/len(loader),          # train_loss
            vloss,                        # val_loss
            metrics['P'], metrics['R'],
            metrics['F1'], metrics['AP']]
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
    torch.save(model.state_dict(), f"checkpoint_stage0-{time.strftime("%Y%m%d-%H%M%S")}.pth")
    print("Stage-0 finished — saved to checkpoint_stage0.pth")
