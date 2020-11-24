import torch
from torch import nn


class SegmentEmbedding(nn.Module):
    def __init__(self, d_embedding, d_model, max_len=300):
        super().__init__()
        self.embedding = nn.Embedding(3, d_embedding) # Segment Embedding
        self.fep_linear = nn.Linear(d_embedding, d_model) # For factorized embedding parameterization (from ALBERT)

    def forward(self, x):
        title_end_ix = torch.where(x == 4)[1][0] # sep token index when title end
        body_end_ix = torch.where(x == 4)[1][1] # sep token index when body end

        segment_title = torch.tensor(0).repeat(title_end_ix + 1) # Include SEP token
        segment_body = torch.tensor(1).repeat(body_end_ix - title_end_ix) # It's already include SEP token because ix is index
        segment_ans = torch.tensor(2).repeat(x.size(1) - (body_end_ix + 1)) # Add 1 becaues it's index

        segment = torch.cat((segment_title, segment_body, segment_ans))
        segment = self.fep_linear(self.embedding(segment))

        return segment.repeat(x.size(0), 1, 1)