import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim=512, heads=8):
        super(MultiheadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads

        assert self.head_dim * heads == embedding_dim, "embedding_dim must be divisible by heads"

        # Linear layers for query, key, and value
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # After we concatenate, fully connected layer to get the output
        # 8*64 = 512 ->512
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, queries, keys, values, mask=None):
        # Number of examples we send at the same time, i.e. batch size
        N = queries.shape[0]
        # Source sequence length or target sequence length
        query_len, key_len, value_len = queries.shape[1], keys.shape[1], values.shape[1]
        
        # Split the single embedding vector into multiple heads
        # (N, seq_len, embedding_dim) -> (N, seq_len, heads, head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        # Scaled-dot product attention (SDPA): multiply the queries with the keys
        # product shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # Add optional mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        sdpa = torch.softmax(energy / self.embedding_dim ** 0.5, dim=-1)
        # Attention output
        sdpa_out = torch.einsum("nhqk,nkhd->nqhd", [sdpa, values])
        # (batch_size, seq_len, embedding_dim)
        sdpa_out = self.fc_out(sdpa_out)

        return sdpa_out