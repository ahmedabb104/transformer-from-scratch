import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        src_pad_idx,
        target_pad_idx,
        device="cuda"
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, device)
        self.decoder = Decoder(target_vocab_size, device)
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        # (N, seq_len) -> (N, 1, 1, seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_target_mask(self, target):
        # (N, seq_len) -> (N, 1, seq_len, seq_len)
        N, target_len = target.shape
        # Expand the lower triangular matrix to the batch size so that each training example has its own mask
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(N, 1, target_len, target_len)
        return target_mask.to(self.device)
    
    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(target, encoder_output, src_mask, target_mask)
        return decoder_output