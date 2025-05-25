import argparse
import yaml
import torch
from pathlib import Path
from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.coco_dataset import COCODataset
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
from utils.geometry import random_warp
from utils.eval import evaluate_detector_pr, sweep_best_thr, evaluate_repeatability


class COCOWarpPairs(torch.utils.data.Dataset):
    """
    Wrap a COCODataset and, at __getitem__,                  

        img          → original image                        
        img_warped   → random homography of img              
        H12          → 3×3 homography that maps img → img_warped
    """
    def __init__(self, coco_ds):
        self.ds = coco_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, *_ = self.ds[idx]                    # img : tensor [3,H,W]
        img_w, H = random_warp(img.unsqueeze(0))  # (1,3,H,W), (1,3,3)
        return img, img_w[0], H[0]


def build_model(cfg, device, ckpt_path: Path | str | None = None) -> torch.nn.Module:
    """Construct the network and (optionally) load a checkpoint."""
    model = SuperPointViT(
        HeadConfig(cfg["model"]["dim_descriptor"]),
        freeze_backbone=True
    ).to(device)

    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        print(f"Loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        print("model is loaded")
        state_dict = state["model"] if isinstance(
            state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"Loaded with missing={len(missing)}  unexpected={len(unexpected)} keys")

    model.eval()
    return model


def evaluate_one_checkpoint(ckpt_path: Path, cfg, device, val_loader):
    model = build_model(cfg, device, ckpt_path)
    best_t, best_F1, best_P, best_R = sweep_best_thr(
        model, val_loader, device=device)

    print(f"\ncheckpoint : {ckpt_path}"
          f"\nbest thr   : {best_t:.2f}"
          f"\nprecision  : {best_P:.3f}"
          f"\nrecall     : {best_R:.3f}"
          f"\nF1         : {best_F1:.3f}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=str,
                        help="Path to Stage-1 checkpoint")
    parser.add_argument("--stage2", type=str,
                        help="Path to Stage-2 checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open("config.yaml"))

    ckpt = args.stage1 if not None else args.stage2

    # dataset
    ps_cfg = PersistentSynthConfig()
    val_loader = torch.utils.data.DataLoader(
        PersistentSyntheticShapes("val", ps_cfg),
        batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     PersistentSyntheticShapes("val", ps_cfg),
    #     batch_size=args.batch_size or cfg["train"]["batch_size"],
    #     shuffle=False,
    #     num_workers=cfg["data"]["num_workers"],
    #     pin_memory=True
    # )

    # model
    model = build_model(cfg, device, ckpt_path=ckpt)
    print("calculating metrics…\n")
    best_t, best_F1, best_P, best_R = sweep_best_thr(model, val_loader, device, verbose=True)
    print(f"checkpoint  {ckpt}\n"
        f"best thr={best_t:.2f}\nP={best_P:.3f}\nR={best_R:.3f}\nF1={best_F1:.3f}")
