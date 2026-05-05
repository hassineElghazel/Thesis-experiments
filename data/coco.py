import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    COCO val2017 image-caption dataset.

    Expects:
        coco_dir/images/val2017/*.jpg
        coco_dir/annotations/captions_val2017.json

    If files are missing, raises FileNotFoundError with download instructions.
    """

    def __init__(
        self,
        coco_dir: str = "data/coco",
        n_images: int = 4000,
        captions_per_image: int = 2,
        img_size: int = 224,
        seed: int = 42,
    ):
        self.img_size = img_size
        ann_path = os.path.join(coco_dir, "annotations", "captions_val2017.json")
        img_dir = os.path.join(coco_dir, "images", "val2017")

        if not os.path.exists(ann_path) or not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"COCO val2017 not found at {coco_dir}.\n"
                "Download:\n"
                "  wget http://images.cocodataset.org/zips/val2017.zip\n"
                "  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
                f"  unzip both into {coco_dir}/"
            )

        with open(ann_path) as f:
            ann_data = json.load(f)

        # Build image_id → captions index
        img_id_to_caps: dict[int, list[str]] = {}
        for ann in ann_data["annotations"]:
            img_id_to_caps.setdefault(ann["image_id"], []).append(ann["caption"])

        # Build image_id → filename index
        img_id_to_file: dict[int, str] = {
            im["id"]: im["file_name"] for im in ann_data["images"]
        }

        rng = random.Random(seed)
        img_ids = [iid for iid in img_id_to_caps if iid in img_id_to_file]
        rng.shuffle(img_ids)
        img_ids = img_ids[:n_images]

        self.samples: list[tuple[str, str]] = []  # (image_path, caption)
        for iid in img_ids:
            caps = img_id_to_caps[iid]
            chosen = rng.sample(caps, min(captions_per_image, len(caps)))
            for cap in chosen:
                self.samples.append((os.path.join(img_dir, img_id_to_file[iid]), cap))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        return img, caption
