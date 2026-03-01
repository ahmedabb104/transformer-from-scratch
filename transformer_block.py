import torch.nn as nn
from multihead_attention import MultiheadAttention

# General transformer block used in the encoder and decoder
# Contains multihead attention and feed forward network
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