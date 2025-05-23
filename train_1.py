import yaml
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_shapes import SyntheticShapes
from utils.losses import focal_bce

cfg = yaml.safe_load(open("configs/dino.yaml"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Data --------------------------------------------------------------------
ds = SyntheticShapes(length=cfg["data"]["synthetic_len"],
                     img_size=cfg["data"]["img_size"])
loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                    shuffle=True, num_workers=cfg["data"]["num_workers"])

# 2. Model / Optimiser -------------------------------------------------------
model = SuperPointViT(HeadConfig(dim_descriptor=cfg["model"]["dim_descriptor"]),
                      freeze_backbone=cfg["model"]["freeze_backbone_stage1"]).to(device)
opt = torch.optim.AdamW([
    {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
     "lr": cfg["train"]["lr_backbone"]},
    {"params": model.det_head.parameters(
    ), "lr": cfg["train"]["lr_heads"]},
])

# 3. Training loop -----------------------------------------------------------
for epoch in range(cfg["train"]["epochs_stage1"]):
    model.train()
    running = 0
    for img, heat_gt in tqdm(loader, desc=f"Epoch {epoch}"):
        img, heat_gt = img.to(device), heat_gt.to(device)
        heat_pred, _ = model(img)
        loss = focal_bce(heat_pred, heat_gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
    print(f"Epoch {epoch} | Loss {running/len(loader):.4f}")
torch.save(model.state_dict(), "checkpoint_stage1.pth")
