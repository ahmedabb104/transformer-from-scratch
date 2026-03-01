import torch
import torch.nn as nn
from multihead_attention import MultiheadAttention
from transformer_block import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(
        self,
        device,
        embedding_dim = 512,
        heads = 8,
        dropout = 0.1,
        forward_expansion = 4
    ):
        super(DecoderBlock, self).__init__()
        self.attention = MultiheadAttention(embedding_dim, heads)
        self.norm = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(embedding_dim, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        device,
        embedding_dim = 512,
        num_layers = 6,
        heads = 8,
        forward_expansion = 4,
        dropout = 0.1,
        max_length = 100
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.word_embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList([DecoderBlock(device, embedding_dim, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_output, src_mask, target_mask)
        out = self.fc_out(x)
        return out