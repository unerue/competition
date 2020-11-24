import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, embs_vocab, hidden_size=64, num_layers=1, 
                 dropout=0., bidirectional=False, num_classes=30):
        super().__init__()
        self.logger = logging.getLogger(__class__.__qualname__)
        self.logger.info('Learning...')
        coef = 2 if bidirectional else 1
        dropout = dropout if num_layers > 1 else 0

        self.embedding = nn.Embedding.from_pretrained(embs_vocab, freeze=True)
        
        self.question = nn.LSTM(
            embs_vocab.size(1), 
            hidden_size,
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout
        )
        
        self.answer = nn.LSTM(
            embs_vocab.size(1), 
            hidden_size,
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size*coef, 64),                  
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, q, a):
        q = self.embedding(q)
        a = self.embedding(a)
        
        q_rnn, _ = self.question(q)
        a_rnn, _ = self.answer(a)
        
        q_rnn, _ = q_rnn.max(dim=0, keepdim=False) 
        a_rnn, _ = a_rnn.max(dim=0, keepdim=False) 
        
        out = torch.cat([q_rnn, a_rnn], dim=-1)
        out = self.classifier(out).sigmoid()
        
        return out


class CNN(nn.Module):
    def __init__(self, embs_vocab, n_filters, filter_sizes, num_classes, 
                 dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embs_vocab, freeze=True)
        
        self.question = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])

        self.answer = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)