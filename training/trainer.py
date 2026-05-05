import torch
import torch.optim as optim
from .losses import infonce


class Trainer:
    """
    Generic InfoNCE trainer for SharedTokenEncoder and DualEncoderFromScratch.
    Both models expose encode_img_tokens / encode_txt_tokens (or the _TokenMLP directly).
    """

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

    def fit_shared(self, encoder, img_toks: torch.Tensor, txt_toks: torch.Tensor) -> list[float]:
        """Train SharedTokenEncoder. Returns per-epoch losses."""
        img_toks = img_toks.to(self.device)
        txt_toks = txt_toks.to(self.device)
        opt = optim.AdamW(encoder.encoder.parameters(),
                          lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg.epochs)
        N = img_toks.size(0)
        losses = []
        encoder.encoder.train()
        for ep in range(self.cfg.epochs):
            perm = torch.randperm(N)
            ep_loss, nb = 0.0, 0
            for i in range(0, N, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                ie = encoder.encoder(img_toks[idx])
                te = encoder.encoder(txt_toks[idx])
                loss = infonce(ie, te, self.cfg.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
            sched.step()
            losses.append(ep_loss / nb)
            if (ep + 1) % 50 == 0:
                print(f"  [SharedEncoder] Epoch {ep+1:>3}/{self.cfg.epochs}  "
                      f"loss = {losses[-1]:.4f}")
        encoder.encoder.eval()
        return losses

    def fit_dual(self, dual_enc, img_toks: torch.Tensor, txt_toks: torch.Tensor) -> list[float]:
        """Train DualEncoderFromScratch. Returns per-epoch losses."""
        img_toks = img_toks.to(self.device)
        txt_toks = txt_toks.to(self.device)
        params = list(dual_enc.img_encoder.parameters()) + list(dual_enc.txt_encoder.parameters())
        opt = optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg.epochs)
        N = img_toks.size(0)
        losses = []
        dual_enc.img_encoder.train()
        dual_enc.txt_encoder.train()
        for ep in range(self.cfg.epochs):
            perm = torch.randperm(N)
            ep_loss, nb = 0.0, 0
            for i in range(0, N, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                ie = dual_enc.encode_img_tokens(img_toks[idx])
                te = dual_enc.encode_txt_tokens(txt_toks[idx])
                loss = infonce(ie, te, self.cfg.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
            sched.step()
            losses.append(ep_loss / nb)
            if (ep + 1) % 50 == 0:
                print(f"  [DualScratch] Epoch {ep+1:>3}/{self.cfg.epochs}  "
                      f"loss = {losses[-1]:.4f}")
        dual_enc.img_encoder.eval()
        dual_enc.txt_encoder.eval()
        return losses

    def fit_tinyclip(self, model, img_tensors: torch.Tensor,
                     tok_ids: torch.Tensor) -> list[float]:
        """Train TinyCLIPFromScratch on pre-computed image tensors and token ids.

        img_tensors stays on CPU (4+ GB for 8k×3×224×224) — moving the whole
        thing to MPS at once stalls Apple's unified-memory allocator. We copy
        per-batch instead. tok_ids is small enough to stay on device.
        """
        img_tensors = img_tensors.cpu().contiguous()
        tok_ids = tok_ids.to(self.device)
        epochs = getattr(self.cfg, "tinyclip_epochs", self.cfg.epochs)
        params = (list(model.image_encoder.parameters())
                  + list(model.text_encoder.parameters()))
        opt = optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        N = img_tensors.size(0)
        losses = []
        model.image_encoder.train()
        model.text_encoder.train()
        for ep in range(epochs):
            perm = torch.randperm(N)
            ep_loss, nb = 0.0, 0
            for i in range(0, N, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                img_batch = img_tensors[idx].to(self.device, non_blocking=True)
                img_emb = model.image_encoder(img_batch)
                txt_emb = model.text_encoder(tok_ids[idx])
                loss = infonce(img_emb, txt_emb, self.cfg.temperature)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ep_loss += loss.item()
                nb += 1
            sched.step()
            losses.append(ep_loss / nb)
            if ep == 0 or (ep + 1) % 5 == 0:
                print(f"  [TinyCLIP] Epoch {ep+1:>3}/{epochs}  "
                      f"loss = {losses[-1]:.4f}")
        model.image_encoder.eval()
        model.text_encoder.eval()
        return losses
