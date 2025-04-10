
##############################
# Training Script (Original Training and Loss Logging)
##############################

# Training parameters.
batch_size = 128
block_size = 100
max_iters = 5000
eval_interval = 500
eval_iters = 32
# These parameters will be varied in the hyperparameter grid below.
default_lr = 3e-4  
default_ce_scale = 10.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
import torch

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 128  # total tokens per chunk (input+target); adjust as needed

def load_data(dataset_dir, dataset_name, split):
    """
    Loads the saved torch tensor from the given dataset directory.
    Expected file names are '<split>.pt' (e.g. train.pt or val.pt).
    """
    path = os.path.join(os.path.expanduser(dataset_dir), dataset_name, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    data = torch.load(path)
    return data

def create_chunks(data, block_size):
    """
    Splits the long tokenized tensor into smaller fixed-size chunks.
    Since we want to create input-target pairs (i.e. shifted by one token),
    each chunk here is of length block_size and will later be split into
    input (first block_size-1 tokens) and target (last block_size-1 tokens).
    """
    chunks = []
    # Step in block_size tokens; you could also use a sliding window if you prefer overlap.
    for i in range(0, len(data) - block_size, block_size):
        chunk = data[i : i + block_size]
        chunks.append(chunk)
    return chunks

def split_input_target(chunk):
    """
    Given a chunk (a tensor of tokens), return a tuple of
    (input_sequence, target_sequence) where target is the input shifted by one token.
    """
    return chunk[:-1], chunk[1:]

class LanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        # Convert each chunk into an (input, target) pair.
        # Note: This computes the pairs on initialization; alternatively, you could
        # compute it on the fly in __getitem__ if memory is a concern.
        self.input_target_pairs = [split_input_target(chunk) for chunk in chunks]

    def __len__(self):
        return len(self.input_target_pairs)

    def __getitem__(self, idx):
        x, y = self.input_target_pairs[idx]
        # Converting the segments to tensor with long type. (They might already be tensors,
        # but this ensures correct dtype if using lists or numpy arrays.)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# Configuration: Adjust these variables or expose them as arguments if needed.
DATASET_DIR = "~/.diffformer/datasets"
DATASET_NAME = "shakespeare"  # or 'rocstories' or 'openwebtext'

# Load previously saved tokenized data
train_data = load_data(DATASET_DIR, DATASET_NAME, "train")
val_data = load_data(DATASET_DIR, DATASET_NAME, "val")

# Create fixed-size chunks from the loaded tensors.
train_chunks = create_chunks(train_data, BLOCK_SIZE)
val_chunks = create_chunks(val_data, BLOCK_SIZE)

  
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])[:, -1]
    return x.to(device), y.to(device)

optimizer = None  # will be created in train_model.

def compute_loss(prefix, target, noise_fn=torch.randn):
    B = prefix.size(0)
    clean_emb = m.token_embedding(target)  # (B, d_model)
    noise = noise_fn(B, m.d_model, device=device)
    t = torch.randint(0, m.num_timesteps, (B,), device=device)
    alpha_bar_t = m.alphas_bar[t].view(B, 1)
    noisy_emb = torch.sqrt(alpha_bar_t) * clean_emb + torch.sqrt(1 - alpha_bar_t) * noise
    _, mse_loss, ce_loss = m(prefix, t, noisy_emb=noisy_emb, noise=noise, targets=target)
    return mse_loss, ce_loss

def estimate_loss():
    losses = {"train": {"total": 0, "mse": 0, "ce": 0},
              "val": {"total": 0, "mse": 0, "ce": 0}}
    m.eval()
    for split in ["train", "val"]:
        total_loss_vals = torch.zeros(eval_iters, device=device)
        mse_loss_vals = torch.zeros(eval_iters, device=device)
        ce_loss_vals = torch.zeros(eval_iters, device=device)
        with torch.no_grad():
            for k in range(eval_iters):
                X, y = get_batch(split)
                mse_loss, ce_loss = compute_loss(X, y)
                total_loss_vals[k] = (mse_loss + ce_loss / default_ce_scale).item()
                mse_loss_vals[k] = mse_loss.item()
                ce_loss_vals[k] = ce_loss.item()
        losses[split]["total"] = total_loss_vals.mean().item()
        losses[split]["mse"] = mse_loss_vals.mean().item()
        losses[split]["ce"] = ce_loss_vals.mean().item()
    m.train()
    return losses

# Logging lists.
train_steps = []
train_total_history = []
train_mse_history = []
train_ce_history = []
val_steps = []
val_total_history = []
val_mse_history = []
val_ce_history = []

##############################
# Training Loop Function (for one set of hyperparameters)
##############################

def train_model(lr, ce_scale, max_iterations):
    global m, optimizer, default_ce_scale
    # Reinitialize the diffusion model for a fair comparison.
    model = DiffusionTransformer(
        vocab_size=len(tokenizer),
        d_model=96,
        num_layers=3,
        num_heads=6,
        block_size=128,
        param_mode="epsilon"
    )
    # Move to device and compile.
    m_local = model.to(device)
    m_local = torch.compile(m_local)
    # Use provided lr.
    optimizer_local = torch.optim.AdamW(m_local.parameters(), lr=lr)

    # Reset logging lists.
    local_train_steps, local_train_total, local_train_mse, local_train_ce = [], [], [], []
    local_val_steps, local_val_total, local_val_mse, local_val_ce = [], [], [], []

    # Use the given ce_scale for this run.
    default_ce_scale = ce_scale

    for step in range(max_iterations):
        if step % eval_interval == 0 or step == max_iterations - 1:
            losses = estimate_loss()
            print(f"[lr={lr}, ce_scale={ce_scale}] step {step}: train loss {losses['train']['total']:.4f} (mse: {losses['train']['mse']:.4f}, ce: {losses['train']['ce']:.4f}), "
                  f"val loss {losses['val']['total']:.4f} (mse: {losses['val']['mse']:.4f}, ce: {losses['val']['ce']:.4f})")
            local_val_steps.append(step)
            local_val_total.append(losses["val"]["total"])
            local_val_mse.append(losses["val"]["mse"])
            local_val_ce.append(losses["val"]["ce"])
        
        X, y = get_batch("train")
        mse_loss, ce_loss = compute_loss(X, y)
        loss = mse_loss + ce_loss / ce_scale

        optimizer_local.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_local.step()

        local_train_steps.append(step)
        local_train_total.append(loss.item())
        local_train_mse.append(mse_loss.item())
        local_train_ce.append(ce_loss.item())

    # Final evaluation.
    with torch.no_grad():
        X_val, y_val = get_batch("val")
        mse_loss, ce_loss = compute_loss(X_val, y_val)
        final_loss = mse_loss + ce_loss / ce_scale
        print(f"[lr={lr}, ce_scale={ce_scale}] Final Eval Loss: {final_loss.item():.4f} "
              f"(mse: {mse_loss.item():.4f}, ce: {ce_loss.item():.4f})")
    return {
        "final_loss": final_loss.item(),
        "train_steps": local_train_steps,
        "train_total": local_train_total,
        "train_mse": local_train_mse,
        "train_ce": local_train_ce,
        "val_steps": local_val_steps,
        "val_total": local_val_total,
        "val_mse": local_val_mse,
        "val_ce": local_val_ce,
    }

##############################
# Hyperparameter Grid Search
##############################

candidate_lrs = [1e-4, 3e-4, 1e-3]
candidate_ce_scales = [5.0, 10.0, 20.0]
results = {}

# To save time, you might run a shorter training run when sweeping hyperparameters.
grid_max_iters = 1000  

for lr in candidate_lrs:
    for ce_scale in candidate_ce_scales:
        print(f"==> Training with lr = {lr}, ce_loss_scale = {ce_scale}")
        res = train_model(lr, ce_scale, grid_max_iters)
        results[(lr, ce_scale)] = res["final_loss"]

print("Hyperparameter grid search results:")
for (lr, ce_scale), final_loss in results.items():
    print(f"lr: {lr}, ce_loss_scale: {ce_scale} -> final val loss: {final_loss:.4f}")
