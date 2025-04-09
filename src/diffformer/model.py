import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

##############################
# (Existing) Model Definitions, Utilities, etc.
##############################

# Main diffusion transformer model (same as your final version)
class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,   # internal transformer dimension
        num_layers,
        num_heads,
        block_size,
        num_timesteps=1000,
        param_mode="epsilon"  # for now, we use "epsilon"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.num_timesteps = num_timesteps
        self.param_mode = param_mode

        # Token embedding and positional encoding.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.time_embedding = SinusoidalPosEmb(d_model)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Noise prediction head.
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Projection from d_model back to logit space.
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Layer normalization for the reverse update output.
        self.proj_ln = nn.LayerNorm(d_model)

        # Diffusion schedule parameters.
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    def forward(self, x, t, noisy_emb, noise=None, targets=None):
        """
        Args:
          x: (B, L) token ids (prefix tokens).
          t: (B,) timesteps.
          noisy_emb: (B, d_model) the noisy embeddings for the final token.
          noise: ground-truth noise (Tensor).
          targets: (B,) target token ids (optional).

        Returns:
          A tuple (pred, mse_loss, ce_loss)
        """
        B, L = x.shape
        # 1. Embed prefix tokens.
        token_emb = self.token_embedding(x)             # (B, L, d_model)
        pos_emb = self.pos_embedding[:, :L, :]             # (1, L, d_model)
        time_emb = self.time_embedding(t).unsqueeze(1).expand(B, L, self.d_model)
        context_emb = token_emb + pos_emb + time_emb        # (B, L, d_model)

        # 2. Add noisy embedding.
        if L < self.pos_embedding.size(1):
            extra_pos = self.pos_embedding[:, L:L+1, :].expand(B, 1, self.d_model)
        else:
            extra_pos = self.pos_embedding[:, -1:, :].expand(B, 1, self.d_model)
        extra_time = self.time_embedding(t).unsqueeze(1)     # (B, 1, d_model)
        noisy_token = noisy_emb.unsqueeze(1) + extra_pos + extra_time  # (B, 1, d_model)
        h = torch.cat([context_emb, noisy_token], dim=1)     # (B, L+1, d_model)

        # 3. Run through transformer.
        seq_len = h.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        h = self.transformer(h.transpose(0, 1), mask=mask).transpose(0, 1)  # (B, L+1, d_model)

        # 4. Predict noise.
        pred = self.noise_predictor(h[:, -1, :])  # (B, d_model)

        mse_loss = None
        ce_loss = None
        if noise is not None:
            mse_loss = F.mse_loss(pred, noise)
        if targets is not None:
            alpha_bar_t = self.alphas_bar[t].view(B, 1)
            sqrt_alpha = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
            x0_pred = (noisy_emb - sqrt_one_minus_alpha * pred) / sqrt_alpha  # (B, d_model)
            # Apply layer norm before projection.
            x0_pred = self.proj_ln(x0_pred)
            pred_logits = self.output_projection(x0_pred)  # (B, vocab_size)
            ce_loss = F.cross_entropy(pred_logits, targets)
        return pred, mse_loss, ce_loss

    def generate(self, context, max_new_tokens=50, diffusion_steps=None, top_k=50, eta=0.0):
        if diffusion_steps is None:
            diffusion_steps = self.num_timesteps

        B = context.size(0)
        device = context.device
        generated = context  # (B, L_context)
        new_t = reparameterize_timesteps(self.alphas_bar, diffusion_steps)
        for _ in range(max_new_tokens):
            x_t = torch.randn(B, self.d_model, device=device)  # candidate in embedding space
            for step in reversed(new_t):
                t_current = torch.full((B,), step, device=device, dtype=torch.long)
                alpha_bar_t = self.alphas_bar[t_current].view(B, 1)
                sqrt_alpha = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
                pred, _, _ = self.forward(generated, t_current, noisy_emb=x_t)
                x0_pred = (x_t - sqrt_one_minus_alpha * pred) / sqrt_alpha
                if step > 0:
                    alpha_bar_prev = self.alphas_bar[t_current - 1].view(B, 1)
                else:
                    alpha_bar_prev = torch.ones(B, 1, device=device)
                sqrt_alpha_prev = torch.sqrt(alpha_bar_prev)
                x_t = sqrt_alpha_prev * x0_pred + torch.sqrt(1 - alpha_bar_prev) * pred + (eta * pred)
            logits = self.output_projection(x_t)
            topk_logits, topk_indices = logits.topk(top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_token = topk_indices.gather(dim=-1, index=sampled_idx)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


