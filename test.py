# from datasets.coco_dataset import COCODataset
# import matplotlib.pyplot as plt
# import yaml
# cfg = yaml.safe_load(open("config.yaml"))
# ds_train = COCODataset("train2017",
#                        coco_root="data/coco_processed/train2017/images",
#                        teacher_root="data/coco_processed/train2017/processed",
#                        img_size=cfg["data"]["img_size"],
#                        patch_size=cfg["data"]["patch_size"],
#                        score_thr=0.1,
#                        max_kp=500)

# img, tgt = ds_train[0]
# plt.imshow(img.permute(1, 2, 0))        # input image
# plt.imshow(tgt["heat"][0], alpha=.4)

import numpy as np
import random
import glob
import pathlib

some_npz = random.choice(
    glob.glob('data/coco_processed/train2017/processed/*.npz'))
d = np.load(some_npz)

print("file :", pathlib.Path(some_npz).name)
print("#pts :", len(d["kps"]))
print("min  :", d["scores"].min() if d["scores"].size else None)
print("max  :", d["scores"].max() if d["scores"].size else None)
print("5 samples :", d["scores"][:5])
