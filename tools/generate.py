import os
import argparse
import json
import torch
import logging
from termcolor import colored
import torch.nn.functional as F
from torchdiffeq import odeint

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

logger = logging.getLogger("diffformer.generate")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(PDMLoggerFormatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

# ---------------------------
# Globals and device
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

# ---------------------------
# Main entry-point for generation
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained FlowTransformer model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-dir', type=str, default='~/.diffformer/models',
                        help='Directory containing model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='model_checkpoint.pt',
                        help='Checkpoint filename')
    parser.add_argument('--dataset-dir', type=str, default='~/.diffformer/datasets',
                        help='Directory containing processed datasets')
    parser.add_argument('--dataset-name', type=str, default='shakespeare',
                        choices=['shakespeare', 'rocstories', 'openwebtext'],
                        help='Base name of the dataset')
    parser.add_argument('--prompt', type=str, default='', help='Initial prompt text')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='Number of tokens/characters to generate')
    parser.add_argument('--char-level', action='store_true',
                        help='Use character-level vocabulary')
    parser.add_argument('--method', type=str, default='rk4', help='ODE integration method')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='Number of solver steps from t=1→0')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=1e-4,
                        help='Absolute tolerance for ODE solver')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    args = parser.parse_args()

    # Load tokenizer or char mapping
    if args.char_level:
        char_map_path = os.path.expanduser(
            os.path.join(args.dataset_dir, f"{args.dataset_name}_char", 'char2idx.json')
        )
        if not os.path.exists(char_map_path):
            logger.error(f"Character mapping not found: {char_map_path}")
            return
        with open(char_map_path, 'r', encoding='utf-8') as f:
            char2idx = json.load(f)
        idx2char = {int(v): k for k, v in char2idx.items()}
        vocab_size = len(char2idx)
        logger.info(f"Using char-level vocab size {vocab_size}, mapping loaded from {char_map_path}")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = len(tokenizer)
        logger.info(f"Using token-level vocab size {vocab_size}")

    # Load model checkpoint
    model_path = os.path.expanduser(os.path.join(args.model_dir, args.checkpoint))
    if not os.path.exists(model_path):
        logger.error(f"Checkpoint not found: {model_path}")
        return
    ckpt = torch.load(model_path, map_location=device)
    if 'hyperparameters' in ckpt:
        hyp = ckpt['hyperparameters']
    elif 'args' in ckpt:
        hyp = ckpt['args']
    else:
        logger.warning("No hyperparameters or args found in checkpoint; using defaults.")
        hyp = {}

    # Instantiate model
    model = FlowTransformer(
        vocab_size=vocab_size,
        d_model=hyp.get('d_model', 96),
        num_layers=hyp.get('num_layers', 3),
        num_heads=hyp.get('num_heads', 6),
        block_size=hyp.get('block_size', 100),
    ).to(device)

    # Load weights (strip compiled prefixes)
    raw_sd = ckpt['model_state_dict']
    new_sd = {}
    for full_key, tensor in raw_sd.items():
        # strip any compiled wrapper prefix (e.g., '_orig_mod.' or '.orig_mod.')
        if 'orig_mod.' in full_key:
            _, stripped = full_key.split('orig_mod.', 1)
        else:
            stripped = full_key
        new_sd[stripped] = tensor
      
    model.load_state_dict(new_sd)

    model.eval()

    # Encode prompt
    if args.char_level:
        prompt_indices = [char2idx.get(ch, 0) for ch in args.prompt]
        generated = torch.tensor([prompt_indices], dtype=torch.long, device=device)
    else:
        generated = tokenizer.encode(args.prompt, return_tensors='pt').to(device)
    B = generated.size(0)

    # Generation loop
    for _ in range(args.max_new_tokens):
        logits_t = torch.randn(B, vocab_size, device=device)
        solver_opts = {}
        if args.method.lower() != 'rk4':
            solver_opts['rtol'] = args.rtol
            solver_opts['atol'] = args.atol
        t_vals = torch.linspace(1.0, 0.0, args.num_steps, device=device)

        def ode_func(t_scalar, x_t):
            t_vec = (t_scalar.expand(B) if isinstance(t_scalar, torch.Tensor)
                     else torch.full((B,), t_scalar, device=device))
            return model(generated, t_vec, x_t)

        x0 = odeint(ode_func, logits_t, t_vals, method=args.method,
                    **solver_opts)[-1]
        topk_logits, topk_idx = x0.topk(args.top_k, dim=-1)
        probs = F.softmax(topk_logits, dim=-1)
        sel = torch.multinomial(probs, num_samples=1)
        next_idx = topk_idx.gather(-1, sel)
        generated = torch.cat([generated, next_idx], dim=1)

    # Decode output
    seq = generated[0].tolist()
    if args.char_level:
        out = ''.join(idx2char.get(i, '') for i in seq)
    else:
        out = tokenizer.decode(seq, skip_special_tokens=True)
    logger.info(f"Generated text:\n{out}")

if __name__ == '__main__':
    main()
