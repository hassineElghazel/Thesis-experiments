import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import AbstractEncoder


class CLIPPretrainedWrapper(AbstractEncoder):
    """
    Wraps openai/clip-vit-base-patch32 and exposes both:
      - final L2-normalised [CLS] embeddings
      - token-level: mean patch/text hidden states projected to shared 512-dim space
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, mode: str = "final", batch_size: int = 32, device: str = "cpu"):
        assert mode in ("final", "token"), f"mode must be 'final' or 'token', got {mode}"
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device(device)
        print(f"Loading {self.MODEL_ID}...")
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model.eval()
        print("  CLIP loaded")

    @property
    def name(self) -> str:
        return f"CLIP Pre-trained ({self.mode})"

    def _extract_batch(self, images, texts):
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        if self.mode == "final":
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        else:
            vis_h = out.vision_model_output.last_hidden_state   # (B, 50, 768)
            txt_h = out.text_model_output.last_hidden_state     # (B, seq, 512)
            patch_mean = vis_h[:, 1:, :].mean(dim=1)
            token_mean = txt_h[:, 1:, :].mean(dim=1)
            ie = self.model.visual_projection(patch_mean)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = self.model.text_projection(token_mean)
            te = te / te.norm(dim=-1, keepdim=True)
        return ie.detach().cpu().numpy(), te.detach().cpu().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        # dummy text needed by processor — we only use image output
        dummy = [""] * len(images)
        return np.concatenate(
            [self._extract_batch(images[i:i+self.batch_size], dummy[i:i+self.batch_size])[0]
             for i in range(0, len(images), self.batch_size)]
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        # dummy image needed by processor
        dummy_img = [Image.new("RGB", (224, 224))] * len(texts)
        return np.concatenate(
            [self._extract_batch(dummy_img[i:i+self.batch_size], texts[i:i+self.batch_size])[1]
             for i in range(0, len(texts), self.batch_size)]
        )

    def encode_paired(
        self, images: list[Image.Image], texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        img_embs, txt_embs = [], []
        for i in range(0, len(images), self.batch_size):
            ie, te = self._extract_batch(images[i:i+self.batch_size], texts[i:i+self.batch_size])
            img_embs.append(ie)
            txt_embs.append(te)
        return np.concatenate(img_embs), np.concatenate(txt_embs)
