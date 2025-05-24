import argparse
import yaml
import torch
from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
from utils.eval import sweep_best_thr

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("config.yaml"))

# dataset
ps_cfg = PersistentSynthConfig()
val_loader = torch.utils.data.DataLoader(
    PersistentSyntheticShapes("val", ps_cfg),
    batch_size=cfg["train"]["batch_size"], shuffle=False,
    num_workers=cfg["data"]["num_workers"], pin_memory=True)

# model
model = SuperPointViT(HeadConfig(
    cfg["model"]["dim_descriptor"]), freeze_backbone=True).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
model.eval()

best_t, best_F1, best_P, best_R = sweep_best_thr(model, val_loader, device)
print(f"checkpoint  {args.ckpt}\n"
      f"best thr={best_t:.2f}  P={best_P:.3f}  R={best_R:.3f}  F1={best_F1:.3f}")
