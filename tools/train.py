import os
import argparse
import json
import torch
import logging
from termcolor import colored
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchviz import make_dot

from flowformer.model import FlowTransformer

# ---------------------------
# Custom pdm–style logging
# ---------------------------
class PDMLoggerFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    ICONS = {
        "DEBUG": "🐞",
        "INFO": "✔",
        "WARNING": "⚠",
        "ERROR": "✖",
        "CRITICAL": "‼",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "white")
        icon = self.ICONS.get(record.levelname, "")
        message = super().format(record)
        return colored(f"{icon} {message}", color)

logger = logging.getLogger("diffformer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(PDMLoggerFormatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

# ---------------------------
# Globals and device
# ---------------------------
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Globals that will be set during training/loading
train_data = None
val_data = None
m = None
ce_scale_global = 0.2
# Will hold vocab size based on token vs char level
dataset_vocab_size = None

# ---------------------------
# Data loading and sampling
# ---------------------------
def load_data(dataset_dir, dataset_name, split):
    path = os.path.join(os.path.expanduser(dataset_dir), dataset_name, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    return torch.load(path)


def get_batch(split, batch_size, block_size):
    global train_data, val_data
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])[:, -1]
    return x.to(device), y.to(device)


def get_alpha_bar_and_dot(model, t):
    T = model.num_timesteps
    t_scaled = t * (T - 1)
    t0 = t_scaled.floor().long().clamp(max=T - 2)
    t1 = t0 + 1
    frac = (t_scaled - t0.float()).unsqueeze(1)
    a0 = model.alphas_bar[t0].view(-1,1)
    a1 = model.alphas_bar[t1].view(-1,1)
    alpha_bar_t = a0 + frac * (a1 - a0)
    d_alpha_bar = (a1 - a0) * (T - 1)
    return alpha_bar_t.clamp(min=1e-8), d_alpha_bar

# ---------------------------
# Loss and evaluation
# ---------------------------
def compute_loss(prefix, target, ce_scale_global=1.0):
    B = prefix.size(0)
    t = torch.rand(B, device=prefix.device)  # [B]

    # Sample noise in latent space
    noise = torch.randn(B, m.d_model, device=prefix.device)

    # Embed target tokens to latent space
    x0 = m.token_embedding(target)  # [B, d_model]

    # Diffusion schedule
    alpha_bar_t = torch.cos((t + 0.008) / (1 + 0.008) * torch.pi / 2) ** 2  # [B]
    alpha_bar_t = alpha_bar_t / alpha_bar_t[0]  # normalize to 1 at t=0
    d_alpha_bar = -torch.pi / 2 * torch.sin(torch.pi * t / (2 * (1 + 0.008))) / (1 + 0.008)  # [B]

    # Interpolate in latent space
    sqrt_ab = alpha_bar_t.sqrt().unsqueeze(1)      # [B, 1]
    sqrt_1mab = (1 - alpha_bar_t).sqrt().unsqueeze(1)  # [B, 1]
    x_t = sqrt_ab * x0 + sqrt_1mab * noise          # [B, d_model]

    logits = m(prefix, t, x_t)  # [B, d_model], [B, vocab]

    # Token prediction loss
    loss = F.cross_entropy(logits, target)       # scalar
    return loss

def estimate_loss(batch_size, block_size, eval_iters):
    losses = {split: 0.0 for split in ["train", "val"]}
    m.eval()
    with torch.no_grad():
        for split in ["train", "val"]:
            vals = []
            for _ in range(eval_iters):
                X, y = get_batch(split, batch_size, block_size)
                loss = compute_loss(X, y)
                vals.append(loss.item())
            losses[split] = sum(vals) / len(vals)
    m.train()
    return losses

# ---------------------------
# Training loop
# ---------------------------
def train_model(lr, max_iters, batch_size, block_size, eval_interval, eval_iters):
    global m
    warmup_steps = 500
    model = FlowTransformer(
        vocab_size=dataset_vocab_size,
        d_model=3 * 32,
        num_layers=3,
        num_heads=6,
        block_size=block_size,
    )
    m = model.to(device)
    # visualize once
    X, y = get_batch("train", batch_size, block_size)
    loss = compute_loss(X, y)
    dot = make_dot(loss, params=dict(m.named_parameters()))
    dot.render("grad_graph", format="png")
    try:
        m = torch.compile(m)
    except Exception as e:
        logger.warning("torch.compile failed: %s", e)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=0.0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            lr_now = scheduler.get_last_lr()[0]
            stats = estimate_loss(batch_size, block_size, eval_iters)
            logger.info(f"[lr={lr_now:.2e}] step {step}: train {stats['train']:.4f}, val {stats['val']:.4f}")
        X, y = get_batch("train", batch_size, block_size)
        loss = compute_loss(X, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        X_val, y_val = get_batch("val", batch_size, block_size)
        final_loss, _, _ = compute_loss(X_val, y_val)
        logger.info(f"Final Eval Loss: {final_loss.item():.4f}")
    return final_loss.item()

# ---------------------------
# Main entry-point
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train FlowTransformer on token or character data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=str, default="~/.diffformer/datasets",
                        help="Directory containing processed datasets.")
    parser.add_argument("--dataset-name", type=str, default="shakespeare",
                        choices=["shakespeare", "rocstories", "openwebtext"],
                        help="Base name of the dataset.")
    parser.add_argument("--char-level", action="store_true",
                        help="Train on character-level dataset instead of token-level.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--block-size", type=int, default=100, help="Sequence length.")
    parser.add_argument("--max-iters", type=int, default=1000, help="Training iterations.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval every N steps.")
    parser.add_argument("--eval-iters", type=int, default=32, help="Iters for eval averaging.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--hp-search", action="store_true",
                        help="Perform hyperparam grid search instead of single run.")
    parser.add_argument("--model-dir", type=str, default="~/.diffformer/models",
                        help="Where to save checkpoints.")
    args = parser.parse_args()

    global train_data, val_data, dataset_vocab_size
    # Determine dataset folder & vocab size
    suffix = "_char" if args.char_level else "_token"
    folder = f"{args.dataset_name}{suffix}"
    data_path = os.path.expanduser(args.dataset_dir)
    if args.char_level:
        mapping_file = os.path.join(data_path, folder, "char2idx.json")
        with open(mapping_file, 'r') as f:
            char2idx = json.load(f)
        dataset_vocab_size = len(char2idx)
        logger.info(f"Using character-level data from '{folder}' with vocab size {dataset_vocab_size}")
    else:
        dataset_vocab_size = len(tokenizer)
        logger.info(f"Using token-level data from '{folder}' with vocab size {dataset_vocab_size}")

    # Load data splits
    logger.info("Loading training data...")
    train_data = load_data(args.dataset_dir, folder, "train")
    logger.info("Loading validation data...")
    val_data = load_data(args.dataset_dir, folder, "val")

    if args.hp_search:
        # grid search
        lrs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        results = {}
        logger.info("Starting hyperparameter grid search...")
        for lr in lrs:
            logger.info(f"Testing lr={lr}")
            loss = train_model(lr, 1500, args.batch_size, args.block_size,
                               args.eval_interval, args.eval_iters)
            results[lr] = loss
        best = min(results, key=results.get)
        logger.info(f"Best LR: {best} -> loss {results[best]:.4f}")
    else:
        logger.info("Starting training...")
        final = train_model(args.lr, args.max_iters, args.batch_size,
                             args.block_size, args.eval_interval, args.eval_iters)
        logger.info("Training complete.")
        # Save checkpoint
        path = os.path.expanduser(args.model_dir)
        os.makedirs(path, exist_ok=True)
        ckpt = os.path.join(path, "model_checkpoint.pt")
        torch.save({'model_state_dict': m.state_dict(), 'args': vars(args)}, ckpt)
        logger.info(f"Saved checkpoint to {ckpt}")

if __name__ == "__main__":
    main()
