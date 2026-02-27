from torch import nn
from multihead_attention import MultiheadAttention
from positional_encoding import PositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim = 512,
        heads = 8,
        dropout = 0.1,
        forward_expansion = 4
    ):
        super(EncoderBlock, self).__init__()
        