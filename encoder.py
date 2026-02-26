from torch import nn
from multihead_attention import MultiheadAttention
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim = 512,
        heads = 8,
        layers = 6,
        dropout = 0.1
    ):
        pass