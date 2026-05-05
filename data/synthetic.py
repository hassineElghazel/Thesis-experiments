import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


CONCEPTS = [
    {"color": (210, 40, 40),   "shape": "circle",    "text": "a red circle on a grey background"},
    {"color": (40, 80, 210),   "shape": "rectangle", "text": "a blue rectangle on a grey background"},
    {"color": (40, 170, 40),   "shape": "triangle",  "text": "a green triangle on a grey background"},
    {"color": (210, 170, 0),   "shape": "diamond",   "text": "a yellow diamond on a grey background"},
    {"color": (160, 40, 210),  "shape": "pentagon",  "text": "a purple pentagon on a grey background"},
    {"color": (0, 190, 190),   "shape": "circle",    "text": "a cyan circle on a grey background"},
    {"color": (210, 110, 0),   "shape": "rectangle", "text": "an orange rectangle on a grey background"},
    {"color": (120, 60, 20),   "shape": "triangle",  "text": "a brown triangle on a grey background"},
    {"color": (210, 50, 130),  "shape": "diamond",   "text": "a pink diamond on a grey background"},
    {"color": (60, 60, 60),    "shape": "pentagon",  "text": "a dark pentagon on a grey background"},
    {"color": (180, 180, 30),  "shape": "circle",    "text": "a gold circle on a grey background"},
    {"color": (30, 30, 120),   "shape": "triangle",  "text": "a navy triangle on a grey background"},
]


def make_image(color_rgb: tuple, shape: str, size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    pad = size // 5
    cx, cy = size // 2, size // 2
    r = size // 2 - pad
    if shape == "circle":
        draw.ellipse([pad, pad, size - pad, size - pad], fill=color_rgb)
    elif shape == "rectangle":
        draw.rectangle([pad, pad, size - pad, size - pad], fill=color_rgb)
    elif shape == "triangle":
        draw.polygon([(cx, pad), (size - pad, size - pad), (pad, size - pad)], fill=color_rgb)
    elif shape == "diamond":
        draw.polygon([(cx, pad), (size - pad, cy), (cx, size - pad), (pad, cy)], fill=color_rgb)
    elif shape == "pentagon":
        pts = [
            (int(cx + r * np.cos(np.radians(a - 90))), int(cy + r * np.sin(np.radians(a - 90))))
            for a in range(0, 360, 72)
        ]
        draw.polygon(pts, fill=color_rgb)
    return img


class SyntheticDataset(Dataset):
    def __init__(self, n_concepts: int = 12, n_variations: int = 30, img_size: int = 224, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.images, self.texts, self.concept_ids = [], [], []
        for cid, c in enumerate(CONCEPTS[:n_concepts]):
            for _ in range(n_variations):
                noise = rng.integers(-20, 20, 3)
                col = tuple(np.clip(np.array(c["color"]) + noise, 10, 245).astype(int))
                self.images.append(make_image(col, c["shape"], size=img_size))
                self.texts.append(c["text"])
                self.concept_ids.append(cid)
        self.concept_ids = np.array(self.concept_ids)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.concept_ids[idx]
