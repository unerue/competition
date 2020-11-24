import torch
import numpy as np
from scipy.stats import spearmanr


def metric_function(p, t):
    score = 0
    for i in range(p.shape[1]):
        score += np.nan_to_num(spearmanr(p[:,i], t[:,i])[0])

    score /= 30
    
    return score

@torch.no_grad()
def valid_function(model, loader, loss_fn):
    y_pred, y_true, train_loss = [], [], []
    for q, a, target in loader:
        outputs = model(q, a)
        loss = loss_fn(outputs, target)
        train_loss.append(loss.item())
        y_true.append(target.detach().cpu().numpy())
        y_pred.append(outputs.detach().cpu().numpy())
        
    train_loss = np.array(train_loss).mean()
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    metric = metric_function(y_pred, y_true)
    
    return train_loss, metric