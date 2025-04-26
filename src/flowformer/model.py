import math
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class TimestepEmbedder(nn.Module):
    def __init__(self, emb_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim, bias=True),
        )

    @staticmethod
    def sinusoid(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        # expects t in [0,1]
        sinu = self.sinusoid(t, self.freq_dim)
        return self.mlp(sinu)

class ODEFunc(nn.Module):
    def __init__(self, transformer, time_embedding, tok_emb, pos_emb, xt_pos, mask, B, Lc):
        super().__init__()
        self.transformer    = transformer
        self.time_embedding = time_embedding
        self.tok_emb        = tok_emb
        self.pos_emb        = pos_emb
        self.xt_pos         = xt_pos
        self.mask           = mask
        self.B              = B
        self.Lc             = Lc

    def forward(self, t_var: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Ensure t_var is [B]
        if t_var.dim() == 0:
            t_var = t_var.repeat(self.B)
        # Time-conditioned prompt context
        te_tok  = self.time_embedding(t_var).unsqueeze(1).expand(self.B, self.Lc, -1)  # [B, Lc, H]
        context = self.tok_emb + self.pos_emb + te_tok                                  # [B, Lc, H]
        # Time-conditioned noisy token embedding
        te_xt   = self.time_embedding(t_var).unsqueeze(1)                                # [B,1,H]
        xt_emb  = v.unsqueeze(1) + self.xt_pos + te_xt                                   # [B,1,H]
        # Transformer step
        h_seq = torch.cat([context, xt_emb], dim=1)                                      # [B, Lc+1, H]
        h     = self.transformer(h_seq.transpose(0,1), mask=self.mask).transpose(0,1)    # [B, Lc+1, H]
        return h[:, -1, :]
        
    # Stub callbacks for torchdiffeq solver compatibility
    def callback_step(self, t0, y0, dt):
        return None
    def callback_accept_step(self, t1, y1, dt):
        return None
    def callback_reject_step(self, t0, y0, dt):
        return None
    def callback_step_adjoint(self, *args, **kwargs):
        return None
    def callback_accept_step_adjoint(self, *args, **kwargs):
        return None
    def callback_reject_step_adjoint(self, *args, **kwargs):
        return None

class FlowTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        block_size: int,
    ):
        super().__init__()
        self.vocab_size    = vocab_size
        self.d_model       = d_model
        self.block_size    = block_size
        self.num_timesteps = 1000

        # diffusion schedule buffers
        T = self.num_timesteps
        t = np.arange(0, T+1, dtype=np.float64) / T
        alpha_bar = np.cos((t + 0.008)/(1+0.008) * np.pi/2)**2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = np.clip(betas, 1e-8, 0.999)
        self.register_buffer('betas',     torch.from_numpy(betas).float())
        self.register_buffer('alphas_bar',torch.from_numpy(alpha_bar[1:]).float())
        self.register_buffer('alphas',    1 - self.betas)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.time_embedding  = TimestepEmbedder(d_model, freq_dim=256)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.token_output_block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, vocab_size)
        )
    def forward(self, x: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        B, L     = x.shape
        device   = x.device
        H        = self.pos_embedding.size(-1)
        Lc       = min(L, self.block_size)

        # Ensure t is per-sample
        if t.dim() == 0:
            t = t.repeat(B)
        t_safe   = t.clamp(min=1e-3)

        # Embed prompt tokens + positions
        tok_emb = self.token_embedding(x)[:, :Lc, :]                          # [B, Lc, H]
        pos_emb = self.pos_embedding[:, :Lc, :].expand(B, Lc, -1)            # [B, Lc, H]

        # Embed noisy token position
        pos_idx = min(Lc, self.block_size - 1)
        xt_pos  = self.pos_embedding[:, pos_idx:pos_idx+1, :].expand(B, 1, -1) # [B, 1, H]

        # Causal mask
        seq_len = Lc + 1
        mask    = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Initial state embedding
        te0 = self.time_embedding(t_safe).unsqueeze(1)                        # [B,1,H]
        v0  = (x_t.unsqueeze(1) + xt_pos + te0).squeeze(1)                    # [B, H]

        # Instantiate ODE function module
        ode_func = ODEFunc(
            transformer=self.transformer,
            time_embedding=self.time_embedding,
            tok_emb=tok_emb,
            pos_emb=pos_emb,
            xt_pos=xt_pos,
            mask=mask,
            B=B,
            Lc=Lc
        )

        # Integrate with RK4 adjoint
        integration_times = torch.tensor([0.0, 1.0], device=device)
        v_path = odeint(
            ode_func,
            v0,
            integration_times,
            method='rk4',
            adjoint_params=list(ode_func.parameters())
        )  # [2, B, H]
        v_final = v_path[-1]  # [B, H]

        # Final transformer for logits
        te_tok_f  = self.time_embedding(t_safe).unsqueeze(1).expand(B, Lc, -1) # [B, Lc, H]
        context_f = tok_emb + pos_emb + te_tok_f                               # [B, Lc, H]
        x0        = v_final.unsqueeze(1)                                       # [B, 1, H]
        h_input   = torch.cat([context_f, x0], dim=1)                           # [B, Lc+1, H]
        h2        = self.token_transformer(h_input.transpose(0,1)).transpose(0,1) # [B, Lc+1, H]
        token_logits = self.token_output_block(h2[:, -1, :])                    # [B, vocab_size]

        return token_logits


    def generate(self, x: Tensor, max_new_tokens: int = 50, method: str = "rk4", top_k: int = 50) -> Tensor:
        B = x.size(0)
        device = x.device
        generated = x

        for _ in range(max_new_tokens):
            xt = torch.randn(B, self.d_model, device=device)

            def ode_func(t_scalar: float, x_t: Tensor) -> Tensor:
                t_vec = torch.full((B,), t_scalar, device=device)
                v_pred, _ = self.forward(generated, t_vec, x_t)
                return -v_pred

            t_vals = torch.tensor([1.0, 0.0], device=device)
            x0_latent = odeint(ode_func, xt, t_vals, method=method)[-1]  # [B, d_model]

            _, logits = self.forward(generated, torch.zeros(B, device=device), x0_latent)
            topk_logits, topk_idx = logits.topk(top_k, dim=-1)  # [B, K]
            probs = F.softmax(topk_logits, dim=-1)
            sel = torch.multinomial(probs, num_samples=1)       # [B, 1]
            next_tok = topk_idx.gather(-1, sel)                 # [B, 1]
            generated = torch.cat([generated, next_tok], dim=1)

        return generated

