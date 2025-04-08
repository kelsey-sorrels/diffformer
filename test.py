
##############################
# (Existing) Final Evaluation and Additional Metrics Evaluation Section
##############################

# (You may choose to train the full-length model with your selected hyperparameters.)
# For demonstration purposes, let's assume the chosen configuration is lr=3e-4, ce_loss_scale=10.0.
final_results = train_model(3e-4, 10.0, max_iters)

with torch.no_grad():
    X_val, y_val = get_batch("val")
    mse_loss, ce_loss = compute_loss(X_val, y_val)
    final_loss = mse_loss + ce_loss / 10.0
    print("Final Diffusion Model Eval Loss:", final_loss.item(),
          f"(mse: {mse_loss.item():.4f}, ce: {ce_loss.item():.4f})")

# 1. Inference Latency.
prompt = get_batch("val")[0][:, :block_size]
num_tokens_to_generate = 100
start_time = time.time()
generated_sample = m.generate(prompt, max_new_tokens=num_tokens_to_generate)
inference_latency = time.time() - start_time
print(f"Inference latency: {inference_latency:.4f} seconds for {num_tokens_to_generate} tokens.")

# 2. BLEU and ROUGE Scores.
import nltk
from nltk.tokenize import word_tokenize
gen_text = tokenizer.decode(generated_sample[0].tolist())
ref_text = text[:len(gen_text)]
gen_tokens = word_tokenize(gen_text)
ref_tokens = word_tokenize(ref_text)
bleu_score = nltk.translate.bleu_score.sentence_bleu([ref_tokens], gen_tokens)
print(f"BLEU score: {bleu_score:.4f}")

def rouge_l_score(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    m_len = len(ref_tokens)
    n_len = len(hyp_tokens)
    table = [[0]*(n_len+1) for _ in range(m_len+1)]
    for i in range(m_len):
        for j in range(n_len):
            if ref_tokens[i] == hyp_tokens[j]:
                table[i+1][j+1] = table[i][j] + 1
            else:
                table[i+1][j+1] = max(table[i+1][j], table[i][j+1])
    lcs = table[m_len][n_len]
    recall = lcs / m_len if m_len > 0 else 0
    precision = lcs / n_len if n_len > 0 else 0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0
    return f1

rouge_l = rouge_l_score(ref_text, gen_text)
print(f"ROUGE-L F1 Score: {rouge_l:.4f}")

# 3. FLOPs and Parameter Count.
try:
    from ptflops import get_model_complexity_info
    macs, params_str = get_model_complexity_info(m, (block_size,), as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs: {macs}, Params: {params_str}")
except ImportError:
    print("ptflops not available, skipping FLOPs computation.")

# 4. Test Perplexity using the Reference Transformer.
def compute_perplexity(model, data, block_size):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(data) - block_size, block_size):
            x = data[i:i+block_size].unsqueeze(0).to(device)
            y = data[i+1:i+block_size+1].unsqueeze(0).to(device)
            logits, loss = model(x, targets=y)
            if loss is not None:
                losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses) if losses else float('inf')
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

ref_model = ReferenceTransformer(
    vocab_size=len(tokenizer),
    d_model=96,
    num_layers=3,
    num_heads=6,
    block_size=128
).to(device)
ref_loss, ref_ppl = compute_perplexity(ref_model, train_data, block_size)
print(f"Reference Transformer: Loss = {ref_loss:.4f}, Perplexity = {ref_ppl:.4f}")

# 5. Log Parameterization Mode.
print(f"Parameterization mode: {m.param_mode}")

##############################
# Plot Loss Curves (for the last full training run)
##############################

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(final_results["train_steps"], final_results["train_total"], label="Train Total Loss")
plt.plot(final_results["train_steps"], final_results["train_mse"], label="Train Noise (MSE) Loss")
plt.plot(final_results["train_steps"], final_results["train_ce"], label="Train CE Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Losses (Every Step)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(final_results["val_steps"], final_results["val_total"], label="Val Total Loss", marker="o", linestyle="-")
plt.plot(final_results["val_steps"], final_results["val_mse"], label="Val Noise (MSE) Loss", marker="o", linestyle="-")
plt.plot(final_results["val_steps"], final_results["val_ce"], label="Val CE Loss", marker="o", linestyle="-")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title(f"Validation Losses (Every {eval_interval} Steps)")
plt.legend()
plt.tight_layout()
plt.show()
