import torch
import torch.nn as nn
from multihead_attention import MultiheadAttention

# General transformer block: contains the multihead attention and feed forward network
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim = 512,
        heads = 8,
        dropout = 0.1,
        forward_expansion = 4
    ):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(embedding_dim, heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        attention = self.attention(queries, keys, values, mask)
        # Add skip connection and attention output in the normalization layer
        x = self.dropout(self.norm1(attention + queries))
        # Feed forward network
        forward = self.feed_forward(x)
        # Second normalization layer
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        device,
        embedding_dim = 512,
        num_layers = 6,
        heads = 8,
        forward_expansion = 4,
        dropout = 0.1,
        max_length = 100
    ):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList([TransformerBlock(embedding_dim, heads, forward_expansion, dropout)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # N examples, seq_length tokens
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        for layer in self.layers:
            # v, k, q are the same in the encoder
            out = layer(out, out, out, mask)
        return out