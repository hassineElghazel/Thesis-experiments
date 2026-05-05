import random
from torch.utils.data import Dataset


class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset loaded from HuggingFace Hub (no local storage required).

    Uses nlphuji/flickr30k — ~31k images with 5 captions each.
    Images are PIL.Image objects resized to img_size × img_size.
    """

    def __init__(
        self,
        split: str = "test",
        n_images: int = 4000,
        captions_per_image: int = 2,
        img_size: int = 224,
        seed: int = 42,
    ):
        from datasets import load_dataset

        print(f"  Loading Flickr30k ({split} split) from HuggingFace Hub...")
        ds = load_dataset("nlphuji/flickr30k", split=split, trust_remote_code=True)

        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[:n_images]

        self.samples: list[tuple] = []
        for idx in indices:
            ex = ds[idx]
            img = ex["image"].convert("RGB").resize((img_size, img_size))
            caps = ex["caption"] if isinstance(ex["caption"], list) else [ex["caption"]]
            chosen = rng.sample(caps, min(captions_per_image, len(caps)))
            for cap in chosen:
                self.samples.append((img, cap))

        print(f"  {len(self.samples)} image-caption pairs loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (PIL.Image, str)
