##############################
# Reference Transformer (for comparison)
##############################

class ReferenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, d_model)
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding_table(pos_ids)  # (1, T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)
        x = self.blocks(x.transpose(0,1)).transpose(0,1)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss