import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, p_dropout: float, n_embeddings_max: int):
        super().__init__()

        self.query = nn.Linear(in_features=d_model, out_features=d_k, bias=False)
        self.key = nn.Linear(in_features=d_model, out_features=d_k, bias=False)
        self.value = nn.Linear(in_features=d_model, out_features=d_v, bias=False)

        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        self.dropout = nn.Dropout(p=p_dropout)

        self.register_buffer('tril', torch.tril(torch.ones(n_embeddings_max, n_embeddings_max)))
        self.scaling = math.sqrt(d_k)

    def forward(self, X: torch.Tensor):
        """
        X: (seq_len, d_model)
        Z: (seq_len, d_v)
        """
        self.seq_len, _ = X.shape
        Q = self.query(X) # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
        K = self.key(X) # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
        V = self.value(X) # (seq_len, d_model) @ (d_model, d_v) = (seq_len, d_v)

        # Scaled dot-product attention
        # (seq_len, d_k) @ (seq_len, d_k)^T = (seq_len, seq_len)
        # E.g., [ hello: [ 0.88 0.12 ... ] my: [ 0.11 0.81 ... ] ... ]
        attention_scores = torch.matmul(Q, K.transpose(dim0=0, dim1=1)) / self.scaling
        
        # Masked self-attention
        attention_scores = attention_scores.masked_fill(self.tril[:self.seq_len, :self.seq_len] == 0, float("-inf"))
        
        # Softmax the score vector for each embedding
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)

        # (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
        Z = torch.matmul(attention_weights, V)
        return Z

class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int, p_dropout: float, n_embeddings_max: int):
        super().__init__()
        self.attention_heads = nn.ModuleList([MaskedSelfAttention(d_model=d_model, d_k=d_k, d_v=d_v, p_dropout=p_dropout, n_embeddings_max=n_embeddings_max) for _ in range(n_heads)])
        self.projection = nn.Linear(in_features=n_heads*d_v, out_features=d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, X: torch.Tensor):
        """
        X: (seq_len, d_model)
        Z: (seq_len, d_model)
        """
        Z_n = torch.cat([attention_head(X) for attention_head in self.attention_heads], dim=1)
        Z = self.projection(Z_n)
        Z = self.dropout(Z)
        return Z
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.Dropout(p=p_dropout)
        )
    
    def forward(self, X: torch.Tensor):
        """
        X: (seq_len, d_model)
        FFN: (seq_len, d_model)
        """
        FFN = self.network(X)
        return FFN

class XformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int, d_ff: int, p_dropout: float, n_embeddings_max: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.self_attention = MultiHeadMaskedSelfAttention(n_heads=n_heads, d_model=d_model, d_k=d_k, d_v=d_v, p_dropout=p_dropout, n_embeddings_max=n_embeddings_max)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, X: torch.Tensor):
        """
        X: (seq_len, d_model)
        Z: (seq_len, d_model)
        """
        out = self.self_attention(X)
        Z_ = self.norm1(X + out)
        out = self.ff(Z_)
        Z = self.norm2(Z_ + out)
        return Z

class XformerDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_decoder_xformer_blocks: int, n_heads: int, d_model: int, d_k: int, d_v: int, d_ff: int, p_dropout: float, n_embeddings_max: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_model)
        
        self.pe = torch.zeros(n_embeddings_max, d_model)
        positions = torch.arange(n_embeddings_max, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(d_model, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.pow(10000, 2/d_model * dimensions)
        self.pe[:, 0::2] = torch.sin(positions / frequencies)
        self.pe[:, 1::2] = torch.cos(positions / frequencies)
        
        self.decoder_xformer_blocks = nn.ModuleList([XformerDecoderBlock(n_heads=n_heads, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff, p_dropout=p_dropout, n_embeddings_max=n_embeddings_max) for _ in range(n_decoder_xformer_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.lin = nn.Linear(in_features=d_model, out_features=n_vocab)
    
    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        """
        tokens: (seq_len)
        """
        X = self.embedding(tokens)
        Z = X + self.pe
        
        for block in self.decoder_xformer_blocks:
            Z = block(Z)
        
        Z = self.norm(Z)
        logits = self.lin(Z)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=-1
            )
            return logits, loss