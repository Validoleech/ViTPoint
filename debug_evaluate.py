import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.coco_dataset import COCODataset
from utils.eval import evaluate_detector_pr, sweep_best_thr

def load_model(ckpt_path: Path, cfg, device):
    """
    Build SuperPointViT and load weights from ckpt_path.
    The ckpt may either be a plain state-dict or a dict
    containing "model" (as in the training code above).
    """
    print("Loading model …")
    model = SuperPointViT(HeadConfig(cfg["model"]["dim_descriptor"]))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  Loaded with missing={len(missing)}  unexpected={len(unexpected)} keys")

    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on COCO-val2017")
    parser.add_argument("--ckpt", required=True, type=Path, help="Path to .pth checkpoint")
    parser.add_argument("--thr", default=0.5, type=float, help="Detection threshold (0-1)")
    parser.add_argument("--cfg", default="config.yaml", type=Path, help="Config file (default: config.yaml)")
    parser.add_argument("--batch", type=int, help="Eval batch size (overrides cfg)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="cuda:0 | cpu  (default: auto)")
    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────────
    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    # dataset / dataloader ----------------------------------------------
    ds_val = COCODataset("val2017",
                         coco_root="data/coco_processed/val2017/images",
                         teacher_root="data/coco_processed/val2017/processed",
                         img_size=cfg["data"]["img_size"],
                         patch_size=cfg["data"]["patch_size"])
    val_loader = DataLoader(ds_val,
                            batch_size=args.batch or cfg["train"]["batch_size"],
                            num_workers=cfg["data"]["num_workers"],
                            shuffle=False, pin_memory=True)

    model = load_model(args.ckpt, cfg, device)

    if 0.0 <= args.thr <= 1.0:
        metrics = evaluate_detector_pr(model, val_loader, thr=args.thr, device=device)
        print(f"\nThreshold {args.thr:4.2f} →  "
              f"P:{metrics['P']:.3f}  R:{metrics['R']:.3f}  "
              f"F1:{metrics['F1']:.3f}  AP:{metrics['AP']:.3f}")
    else:
        print("  Sweeping thresholds to find best F1 …")
        best_t, best_f1, P, R = sweep_best_thr(model, val_loader, device=device)
        print(f"\nBest threshold {best_t:4.2f}  "
              f"P:{P:.3f}  R:{R:.3f}  F1:{best_f1:.3f}")

if __name__ == "__main__":
    main()
