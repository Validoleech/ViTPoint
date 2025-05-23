import csv
import yaml
import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
from utils.loss import focal_bce, compute_val_loss
from utils.eval import evaluate_detector_pr, val_step_stage01

if __name__ == "__main__":

    cfg = yaml.safe_load(open("config.yaml"))
    out_csv = "data/stage1_metrics.csv"
    if not os.path.isfile(out_csv):
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "P", "R", "F1", "AP"])
    else:
        raise FileExistsError

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    ps_cfg = PersistentSynthConfig()
    ds_train = PersistentSyntheticShapes("train", ps_cfg)
    ds_val = PersistentSyntheticShapes("val",   ps_cfg)
    
    loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"])

    # Model / Optimiser
    model = SuperPointViT(HeadConfig(dim_descriptor=cfg["model"]["dim_descriptor"]),
                        freeze_backbone=cfg["model"]["freeze_backbone_stage1"]).to(device)
    opt = torch.optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
        "lr": cfg["train"]["lr_backbone"]},
        {"params": model.det_head.parameters(
        ), "lr": cfg["train"]["lr_heads"]},
    ])

    EPOCHS = cfg.get("epochs_stage1", 5)
    for ep in range(EPOCHS):
        model.train()
        running = 0
        for img, heat, off in tqdm(loader, desc=f"E{ep}"):
            img, heat = img.to(device, non_blocking=True), heat.to(
                device, non_blocking=True)
            heat_pred, _ = model(img)
            loss = focal_bce(heat_pred, heat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        print(f"Epoch {ep} | Loss {running/len(loader):.4f}")
        metrics = evaluate_detector_pr(model, val_loader, device=device)
        val_loss = compute_val_loss(model, val_loader, val_step_stage01, device)
        print(f"  Val  P:{metrics['P']:.3f}  R:{metrics['R']:.3f} "
            f"F1:{metrics['F1']:.3f}  AP:{metrics['AP']:.3f}")
        print(f"  Val-loss: {val_loss:.4f}")
        row = [ep,
            running/len(loader),          # train_loss
            val_loss,                     # val_loss
            metrics['P'], metrics['R'],
            metrics['F1'], metrics['AP']]
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
    torch.save(model.state_dict(), "checkpoint_stage1.pth")
