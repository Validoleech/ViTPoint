import yaml
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.coco_dataset import COCODataset
from utils.geometry import random_warp, warp_offsets
from utils.loss import soft_nms_penalty, compute_val_loss, infonce_loss
from utils.eval import evaluate_detector_pr, val_step_stage2, log_grad_norm, calc_ap

if __name__ == "__main__":

    writer = SummaryWriter(log_dir="runs/stage1")
    global_step = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_stage0", required=None,
                        help="Path to stage0 checkpoint. "
                        "If omitted, backbone is initialized with vanilla DINOv2 weights.")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Number of mini-batches to accumulate before every "
                        "optimiser step (1 ⇒ no accumulation)")
    parser.add_argument("--val_interval", type=int, default=4)
    parser.add_argument("--teacher", action="store_false",
                        help="Use ORB keypoints for teaching.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    ds_train = COCODataset("train2017",
                           coco_root="data/coco_processed/train2017/images",
                           teacher_root="data/coco_processed/train2017/processed" if args.teacher else None,
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

    head_cfg = HeadConfig(cfg["model"]["dim_descriptor"],
               subgrid=cfg["model"]["subgrid"])
    model = SuperPointViT(head_cfg, freeze_backbone=True).to(device)
    # unfreeze last 2 ViT blocks
    for blk in model.backbone.blocks[-2:]:
        for p in blk.parameters():
            p.requires_grad = True

    # load stage-0 weights
    if args.ckpt_stage0 is not None and Path(args.ckpt_stage0).is_file():
        model.load_state_dict(torch.load(args.ckpt_stage0,
                                        map_location="cuda:0"), strict=False)
        print(f"Loaded stage-0 weights from {args.ckpt_stage0}")
    else:
        print("Using vanilla DINOv2 weights as backbone")


    # optimiser
    heads = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("backbone.blocks")]
    bb_ft = [p for n, p in model.named_parameters()
             if p.requires_grad and n.startswith("backbone.blocks")]

    lr_heads = float(cfg["train"]["lr_heads"])
    lr_backbone = float(cfg["train"]["lr_backbone"])
    for p in heads:
        p.initial_lr = lr_heads
    for p in bb_ft:
        p.initial_lr = lr_backbone

    opt = torch.optim.AdamW(
        [{"params": heads, "lr": lr_heads},
         {"params": bb_ft, "lr": lr_backbone}],
        weight_decay=1e-4)

    warmup_steps = 1_000
    total_steps = len(loader) * (args.epochs or cfg["train"]["epochs_stage1"])
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps)
    scaler = GradScaler()

    # EMA of parameters
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=None)

    start_ep = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        opt.load_state_dict(ckpt['optim'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep = ckpt['epoch'] + 1
        ema.load_state_dict(ckpt['ema'])
        global_step = ckpt['step']
        print("Resuming from {args.resume} @ epoch {start_ep}")

    L_DICE = 1.0
    L_BCE = 1.5

    L_NMS = 0.05
    L_OFF = 0.5
    L_DESC = 1.0
    L_CONS = 0.5
    GAMMA = 2.5

    EPOCHS = args.epochs or cfg["train"]["epochs_stage1"]

    for ep in range(start_ep, EPOCHS):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"E{ep}"):
            img, *_ = batch
            img = img.to(device, non_blocking=True)
            # heat_gt = heat_gt.to(device, non_blocking=True)
            # off_gt = off_gt.to(device, non_blocking=True)

            img_w, H = random_warp(img)  # img_w ∈ [0,1]
            H = H.to(device)

            with autocast(device_type=device.type, dtype=amp_dtype):
                h0, off0, d0 = model(img)
                d0 = F.normalize(d0, p=2, dim=1)
                h1, off1, d1 = model(img_w)
                d1 = F.normalize(d1, p=2, dim=1)

                h0_warp = warp_offsets(h0, H)
                off0_warp = warp_offsets(off0, H)
                loss_cons_heat = F.l1_loss(h0_warp,   h1)   + F.l1_loss(warp_offsets(h1,   torch.linalg.inv(H)), h0)
                loss_cons_off  = F.l1_loss(off0_warp, off1) + F.l1_loss(warp_offsets(off1, torch.linalg.inv(H)), off0)
                loss_desc = L_DESC * infonce_loss(d0, d1, H)
                loss_nms= 0.5 * (soft_nms_penalty(h0) + soft_nms_penalty(h1))
                loss = (L_CONS * (loss_cons_heat + L_OFF*loss_cons_off) + L_NMS  * loss_nms + L_DESC * loss_desc)

            scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in opt.param_groups:
                    pg['lr'] = pg['initial_lr'] * lr_scale
            else:
                cosine_sched.step()
                
            ema.update_parameters(model)
            
            global_step += 1
            pos_frac = (h0 > 0.5).float().mean()

            writer.add_scalar("train/loss_total",       loss.item(),           global_step)
            writer.add_scalar("train/consistency_heat", loss_cons_heat.item(), global_step)
            writer.add_scalar("train/consistency_off",  loss_cons_off.item(),  global_step)
            writer.add_scalar("train/desc_InfoNCE",     loss_desc.item(),      global_step)
            if global_step % 100 == 0:
                writer.add_scalar("debug/pos_px_frac",  pos_frac,                global_step)
                for i, pg in enumerate(opt.param_groups):
                    writer.add_scalar(f"lr/group{i}", pg['lr'], global_step)
                log_grad_norm(model, writer, "grad_norm/all", global_step)

        avg_loss = running / len(loader)
        print(f"Epoch {ep}  |  avg_loss={avg_loss:.4f}")

        if (ep % args.val_interval) == 0:
            model.eval()
            metrics = evaluate_detector_pr(model, val_loader, device=device)
            vloss = compute_val_loss(model, val_loader,
                                     val_step_stage2, device)

            rep, ap = calc_ap(model, val_loader, device)
            writer.add_scalar("val/P",          metrics['P'],  global_step)
            writer.add_scalar("val/R",          metrics['R'],  global_step)
            writer.add_scalar("val/F1",         metrics['F1'], global_step)
            writer.add_scalar("val/AP",         metrics['AP'], global_step)
            writer.add_scalar("val/loss",       vloss,         global_step)
            writer.add_scalar("val/repeatability", rep,        global_step)
            writer.add_scalar("val/desc_AP",       ap,         global_step)

            print(f"  Val  P:{metrics['P']:.3f}  R:{metrics['R']:.3f} "
                  f"F1:{metrics['F1']:.3f}  AP:{metrics['AP']:.3f}  "
                  f"repeat:{rep:.3f}  descAP:{ap:.3f}")

        ckpt = dict(epoch=ep, step=global_step,
                    model=model.state_dict(),
                    optim=opt.state_dict(),
                    scaler=scaler.state_dict(),
                    ema=ema.state_dict())
        fn = f"checkpoint_stage1-{time.strftime('%Y%m%d-%H%M%S')}-ep{ep}.pth"
        torch.save(ckpt, fn)
        ema_fn = fn.replace(".pth", "_EMA.pth")
        torch.save(ema.state_dict(), ema_fn)

    final_name = f"checkpoint_stage1-{time.strftime('%Y%m%d-%H%M%S')}.pth"
    torch.save(ema.state_dict(), final_name)
    torch.save(model.state_dict(),f"checkpoint_stage1-{time.strftime("%Y%m%d-%H%M%S")}.pth")
    print(f"Stage-1 finished — saved to checkpoint_stage1-{time.strftime("%Y%m%d-%H%M%S")}.pth")
