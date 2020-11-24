import torch
from torch import nn
from .token import TokenEmbedding
from .positional import PositionalEmbedding
from .segment import SegmentEmbedding

class CustomEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix (with Word2Vec)
    2. PositionalEmbedding : adding positional information
    3. SegmentEmbedding : adding segment information (Title, Body, Answer)
    sum of all these features are output of Embedding
    """
    def __init__(self, vocab_size, d_embedding, d_model, pad_idx=0, max_len=300):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=d_embedding, pad_idx=pad_idx)
        self.position = PositionalEmbedding(d_embedding=d_embedding, d_model=d_model, max_len=max_len)
        self.segment = SegmentEmbedding(d_embedding=d_embedding, d_model=d_model)

        self.linear_layer = nn.Linear(d_embedding, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence):
        x = self.linear_layer(self.token(sequence)) + self.position(sequence) + self.segment(sequence)
        return self.norm(x)