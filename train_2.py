import csv
import yaml
import os
import torch
import kornia as K
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vitpoint import SuperPointViT, HeadConfig
from datasets.coco_homography import COCOWarped
from datasets.hpatches_pairs import HPatchesPairs
from utils.loss import focal_bce, info_nce, compute_val_loss
from utils.eval import evaluate_repeatability, val_step_stage2

cfg = yaml.safe_load(open("config.yaml"))
out_csv = "data/stage2_metrics.csv"

# write header once
if not os.path.isfile(out_csv):
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "repeatability"])
else:
    raise FileExistsError


device = "cuda" if torch.cuda.is_available() else "cpu"

hp_ds = HPatchesPairs(root="data/hpatches",
                      img_size=cfg["data"]["img_size"])
ds = COCOWarped(cfg["data"]["coco_root"], img_size=cfg["data"]["img_size"])

loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                    shuffle=True, num_workers=cfg["data"]["num_workers"])
hp_loader = DataLoader(hp_ds, batch_size=1, shuffle=False, num_workers=4)

model = SuperPointViT(HeadConfig(
    cfg["model"]["dim_descriptor"]), freeze_backbone=False).to(device)
model.load_state_dict(torch.load("checkpoint_stage1.pth"), strict=False)

opt = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
    {"params": list(model.det_head.parameters()) +
     list(model.desc_head.parameters()),
     "lr": cfg["train"]["lr_heads"]},
], weight_decay=1e-4)

for epoch in range(cfg["train"]["epochs_stage2"]):
    running = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        imgs, H_mat, warp_grid = batch
        imgs = imgs.to(device)
        H_mat = H_mat.to(device)
        warp_grid = warp_grid.to(device)

        # forward
        heat1, desc1 = model(imgs)                               # original
        imgs_w = K.geometry.transform.resize(
            imgs, imgs.shape[-2:])  # dummy (no resize)
        imgs_w = K.geometry.warp_perspective(
            imgs_w, H_mat, dsize=imgs.shape[-2:])
        heat2, desc2 = model(imgs_w)                              # warped

        # Detector consistency
        heat1_w = K.geometry.warp_perspective(
            heat1, H_mat, dsize=heat1.shape[-2:])
        L_det = focal_bce(heat1_w, heat2) + focal_bce(heat2, heat1_w)

        # Descriptor InfoNCE
        desc1_w = K.geometry.warp_perspective(
            desc1, H_mat, dsize=desc1.shape[-2:])
        L_desc = info_nce(desc1_w, desc2)

        loss = L_det + 0.1 * L_desc

        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
    print(f"Epoch {epoch} | loss {running/len(loader):.4f}")

    # validation
    rep = evaluate_repeatability(model, hp_loader, device=device)
    print(f"  HPatches Repeatability: {rep['repeatability']*100:.2f}%")
    
    val_loss = compute_val_loss(model, loader, val_step_stage2, device)
    print(f"  Val-loss (self-sup): {val_loss:.4f}")
    row = [epoch,
           running/len(loader),          # train_loss
           val_loss,                     # val_loss
           rep['repeatability']]
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow(row)

torch.save(model.state_dict(), "checkpoint_stage2.pth")
