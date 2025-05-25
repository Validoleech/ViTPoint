import yaml
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.coco_dataset import COCODataset
from utils.geometry import random_warp
from utils.loss import focal_bce, l1_offset, soft_nms_penalty, compute_val_loss
from utils.loss import infonce_loss
from utils.eval import evaluate_detector_pr, val_step_stage2

if __name__ == "__main__":

    writer = SummaryWriter(log_dir="runs/stage1")
    global_step = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_stage0", required=True)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    ds_train = COCODataset("train2017",
                           coco_root="data/coco_processed/train2017/images",
                           teacher_root="data/coco_processed/train2017/processed",
                           img_size=cfg["data"]["img_size"],
                           patch_size=cfg["data"]["patch_size"],
                           score_thr=0.1,
                           max_kp=500)
    ds_val = COCODataset("val2017",
                         coco_root="data/coco_processed/val2017/images",
                         teacher_root="data/coco_processed/val2017/processed")

    loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"],
                        num_workers=cfg["data"]["num_workers"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(ds_val,   batch_size=cfg["train"]["batch_size"],
                            num_workers=cfg["data"]["num_workers"], shuffle=False, pin_memory=True)

    # ─ model
    model = SuperPointViT(HeadConfig(cfg["model"]["dim_descriptor"]),
                          freeze_backbone=True).to(device)
    # unfreeze last 2 ViT blocks
    for blk in model.backbone.blocks[-2:]:
        for p in blk.parameters():
            p.requires_grad = True

    # load stage-0 weights
    model.load_state_dict(torch.load(args.ckpt_stage0,
                                     map_location="cpu"), strict=False)

    # ─ optimiser
    heads = [p for n, p in model.named_parameters() if p.requires_grad and
             not n.startswith("backbone.blocks")]
    bb_ft = [p for n, p in model.named_parameters() if n.startswith(
        "backbone.blocks") and p.requires_grad]

    opt = torch.optim.AdamW(
        [{"params": heads, "lr": float(cfg["train"]["lr_heads"])},
         {"params": bb_ft, "lr": float(cfg["train"]["lr_backbone"])}],
        weight_decay=1e-4)

    scaler = GradScaler()

    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        opt.load_state_dict(ckpt['optim'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep = ckpt['epoch'] + 1
        global_step = ckpt['step']
    else:
        start_ep = 0

    A = 0.25
    G = 2.5
    L_NMS = 0.15
    L_OFF = 2.0
    L_DESC = 1.0
    EPOCHS = args.epochs or cfg["train"]["epochs_stage1"]

    for ep in range(start_ep, EPOCHS):
        model.train()
        running = 0
        for img, heat, off in tqdm(loader, desc=f"E{ep}", ncols=90):
            img = img.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True)
            off = off.to(device, non_blocking=True)

            # build warped pair
            img_w, H = random_warp(img)
            H = H.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype):
                h0, off0, d0 = model(img)
                d0 = d0.detach() / d0.norm(dim=1, keepdim=True)
                h1, off1, d1 = model(img_w)
                d1 = d1.detach() / d1.norm(dim=1, keepdim=True)

                loss_det = (focal_bce(h0, heat, A, G) +
                            focal_bce(h1, heat, A, G))/2
                loss_off = (l1_offset(off0, off, heat) +
                            l1_offset(off1, off, heat))/2
                loss_nms = (soft_nms_penalty(h0)+soft_nms_penalty(h1))/2
                loss_desc = infonce_loss(d0, d1, H)

                loss = loss_det + L_OFF*loss_off + L_NMS*loss_nms + L_DESC*loss_desc

            writer.add_scalar("train/loss_det",   loss_det.item(),  global_step)
            writer.add_scalar("train/loss_desc",  loss_desc.item(), global_step)
            writer.add_scalar("train/loss_off",   loss_off.item(),  global_step)
            writer.add_scalar("train/loss_nms",   loss_nms.item(),  global_step)
            writer.add_scalar("train/loss_total", loss.item(),      global_step)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1

            # quick TB
            if global_step % 200 == 0:
                writer.add_scalar("train/loss_total", loss.item(), global_step)

        # ─ validation
        metrics = evaluate_detector_pr(model, val_loader, device=device)
        vloss = compute_val_loss(model, val_loader, val_step_stage2, device)
        print(f"Epoch {ep}")
        print(f"  Val P:{metrics['P']:.3f} R:{metrics['R']:.3f} "
               f"F1:{metrics['F1']:.3f}")
        torch.save({"epoch": ep, "step": global_step,
                    "model": model.state_dict(), "optim": opt.state_dict(),
                    "scaler": scaler.state_dict()},
                   f"checkpoint_stage1-{time.strftime("%Y%m%d-%H%M%S")}-ep{ep}.pth")
    torch.save(model.state_dict(),f"checkpoint_stage1-{time.strftime("%Y%m%d-%H%M%S")}.pth")
    print(f"Stage-1 finished — saved to checkpoint_stage1-{time.strftime("%Y%m%d-%H%M%S")}.pth")
