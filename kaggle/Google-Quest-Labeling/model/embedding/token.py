from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, pad_idx=0):
        super().__init__(vocab_size, embed_size, padding_idx=pad_idx)