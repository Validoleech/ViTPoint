# ViTPoint

## Steps

- Run `datasets/synthetic_dataset_generate.py --force` once from CLI
- Run train_0.py
- Run train_1.py

## Model Description

This project implements a Vision Transformer (ViT) model for point detection. The architecture consists of:

1. Image feature extraction using ViT
2. Two-stage training process (detector on synthetic dataset + detector fine-tune on COCO dataset + descriptor learing)
3. Point detection head.
