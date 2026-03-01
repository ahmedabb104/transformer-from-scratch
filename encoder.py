import torch
import torch.nn as nn
from transformer_block import TransformerBlock

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