import logging
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_embedding, d_model, max_len=300):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.max_len = max_len * 3
        # print(self.max_len)
        self.logger.info(self.max_len)

        self.embedding = nn.Embedding(self.max_len, d_embedding)
        self.fep_linear = nn.Linear(d_embedding, d_model) # For factorized embedding parameterization (from ALBERT)

    def forward(self, x):
        position = torch.arange(x.size(1))
        position = self.fep_linear(self.embedding(position))
        
        return position.repeat(x.size(0), 1, 1)