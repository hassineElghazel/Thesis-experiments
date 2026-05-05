import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


class VQVAETrainer:
    """Trains a VQVAE on an image dataset (COCO train2017)."""

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.transform = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def fit(self, model, image_list: list, batch_size: int = 32) -> list[float]:
        """
        Trains VQ-VAE on a list of PIL Images.
        Returns per-epoch total losses.
        """
        from torch.utils.data import TensorDataset

        print(f"  Pre-processing {len(image_list)} images...")
        tensors = torch.stack([self.transform(img) for img in image_list])

        dataset = torch.utils.data.TensorDataset(tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

        opt = optim.Adam(model.parameters(), lr=self.cfg.vqvae_lr)
        model.to(self.device)
        losses = []

        for ep in range(self.cfg.vqvae_epochs):
            ep_loss, ep_vq, nb = 0.0, 0.0, 0
            for (x,) in loader:
                x = x.to(self.device)
                _, total_loss, vq_loss = model(x)
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                ep_loss += total_loss.item()
                ep_vq += vq_loss.item()
                nb += 1
            avg = ep_loss / nb
            losses.append(avg)
            print(f"  [VQ-VAE] Epoch {ep+1:>2}/{self.cfg.vqvae_epochs}  "
                  f"loss={avg:.4f}  vq={ep_vq/nb:.4f}")

        # Codebook utilisation check
        model.eval()
        all_indices = []
        with torch.no_grad():
            for (x,) in loader:
                idx = model.encode_to_indices(x.to(self.device))
                all_indices.append(idx.cpu())
        all_idx = torch.cat(all_indices)
        util = model.vq.utilisation(all_idx)
        print(f"  Codebook utilisation: {util*100:.1f}% "
              f"({'OK' if util >= 0.7 else 'WARNING — codebook collapse?'})")

        return losses
