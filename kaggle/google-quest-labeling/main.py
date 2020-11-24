import os
import sys
import logging
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold, KFold
from tabulate import tabulate
from torchtext import data

# Custom packages
from models import LSTM
from models.utils import get_dataset, wrapper, splits_cv
from models.utils import metric_function, valid_function


logging.basicConfig(level=logging.INFO)

# if windows -> torchtext/utils.py -> csv.field_size_limit(sys.maxsize)
# print(sys.maxsize)
parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42, required=False)
parser.add_argument('--batch_size', type=int, default=64, required=False)
parser.add_argument('--n_splits', type=int, default=3, required=False)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

ROOT = 'input/google-quest-challenge/'
train = pd.read_csv(ROOT + 'train.csv')
labels = train.columns[11:]
del train

train_ds, test_ds, emb_vocab = get_dataset(ROOT)
print('Embedding vocab size:', emb_vocab.size()[0])

vocab_size = emb_vocab.size()[0]
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cv = KFold(n_splits=args.n_splits, random_state=args.seed)

test_loader = wrapper(test_ds, batch_size=args.batch_size, shuffle=False, repeat=False, labels=labels, mode='test')

# formatter = '{:d}-fold Epoch {:d}({:.2f}s): \n\t- Train loss {:.3f} Spearman {:.3f}\n\t- Valid loss {:.3f} Spearman {:.3f}\n'
headers = ['Fold', 'Epoch', 'Time(s)', 'Train loss', 'Train Spearman', 'Valid loss', 'Valid Spearman']

def oof_preds(train_ds, test_loader, embs_vocab, epochs=3):
    logger = logging.getLogger(__name__)
    table = []
    for num_cv, (loader, vloader) in enumerate(splits_cv(train_ds, cv, batch_size=args.batch_size)):
        model = LSTM(embs_vocab, hidden_size=128, dropout=0.1, bidirectional=True).to(device)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 1e-3, betas=(0.75, 0.999), weight_decay=1e-3)
        loss_fn = torch.nn.BCELoss()

        for epoch in range(1, epochs+1):
            y_pred, y_true = [], []
            start_time = time.time()
            tloss = []          
            model.train()
            
            for q, a, target in loader:
                optimizer.zero_grad()
                outputs = model(q, a)
                loss = loss_fn(outputs, target)
                tloss.append(loss.item())
                loss.backward()
                optimizer.step()
                y_true.append(target.detach().cpu().numpy())
                y_pred.append(outputs.detach().cpu().numpy())

            tloss = np.array(tloss).mean()
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            tmetric = metric_function(y_pred, y_true)

            vloss, vmetric = valid_function(model, vloader, loss_fn)
            if epoch % 1 == 0:
                table.append([num_cv+1, epoch, ((time.time() - start_time) / 60), tloss, tmetric, vloss, vmetric])
                logger.info(f'{num_cv+1}-fold Epoch {epoch} Train loss {tloss}')
                # print(formatter.format(num_cv+1, epoch, ((time.time() - start_time) / 60), tloss, tmetric, vloss, vmetric))

        print(tabulate(table, headers=headers))

        # qa_id, preds = [], [] 
        # with torch.no_grad():
        #     for qaids, q, a in test_loader:
        #         outputs = model(q, a)
        #         qa_id.append(qaids.cpu().numpy())
        #         preds.append(outputs.detach().cpu().numpy())
        
        # qa_id = np.concatenate(qa_id)
        # preds = np.concatenate(preds)
        # sub.loc[qa_id, labels]  =  sub.loc[qa_id, labels].values + preds / args.n_splits


oof_preds(train_ds, test_loader, emb_vocab, epochs=3)


# sub.to_csv('submission.csv')