import yaml
import torch
import kornia as K
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vitpoint import SuperPointViT, HeadConfig
from datasets.coco_homography import COCOWarped
from utils.loss import focal_bce, info_nce

cfg = yaml.safe_load(open("config.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = COCOWarped(cfg["data"]["coco_root"], img_size=cfg["data"]["img_size"])
loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                    shuffle=True, num_workers=cfg["data"]["num_workers"])

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

        # forward ----------------------------------------------------------------
        heat1, desc1 = model(imgs)                               # original
        imgs_w = K.geometry.transform.resize(
            imgs, imgs.shape[-2:])  # dummy (no resize)
        imgs_w = K.geometry.warp_perspective(
            imgs_w, H_mat, dsize=imgs.shape[-2:])
        heat2, desc2 = model(imgs_w)                              # warped

        # 1) Detector consistency
        heat1_w = K.geometry.warp_perspective(
            heat1, H_mat, dsize=heat1.shape[-2:])
        L_det = focal_bce(heat1_w, heat2) + focal_bce(heat2, heat1_w)

        # 2) Descriptor InfoNCE
        desc1_w = K.geometry.warp_perspective(
            desc1, H_mat, dsize=desc1.shape[-2:])
        L_desc = info_nce(desc1_w, desc2)

        loss = L_det + 0.1 * L_desc

        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
    print(f"Epoch {epoch} | loss {running/len(loader):.4f}")

torch.save(model.state_dict(), "checkpoint_stage2.pth")
