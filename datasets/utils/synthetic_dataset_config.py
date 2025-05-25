from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PersistentSynthConfig:
    """
    Configuration for synthetic image generation, including dataset paths,
    primitive types, split sizes, image resolutions, and augmentation parameters.
    """
    data_root: str = "data/synthetic"
    primitives: List[str] = field(default_factory=lambda: [
        "lines",
        "polygon",
        "multiple_polygons",
        "ellipses",
        "star",
        "checkerboard",
        "stripes",
        "cube",
        "random_cubes",
        "gaussian_noise"
    ])
    # how many images per primitive & split
    split_sizes: Dict[str, int] = field(default_factory=lambda:
                                        {"train": 30_000, "val":  500, "test": 1_000})
                                        # {"train": 10, "val":  1, "test": 1}) # debug
    gen_img_size: tuple = (960, 1280)           # generation resolution
    resize: tuple = (224, 224)                  # net input
    patch_size: int = 14                        # ViT-S/14
    blur: int = 5                               # Gaussian blur (odd -> enabled)
    random_seed: int = 0
