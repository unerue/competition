import os
import logging
import argparse
import pandas as pd 
import numpy as np
import lightgbm as lgb
from models.data_loader import data_loader, data_loader_all
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss




train_path = 'input/train'
test_path = 'input/test'
label = pd.read_csv('input/train_label.csv')
sub = pd.read_csv('input/sample_submission.csv')

train = data_loader_all(data_loader, 
                        path=train_path, 
                        train=True, 
                        nrows=100, # 600
                        normal=999, 
                        event_time=10, 
                        lookup_table=label)

test = data_loader_all(data_loader, 
                       path=test_path, 
                       train=False, 
                       nrows=60)

print(train.shape, test.shape)

train = train.fillna(0)
test = test.fillna(0)
test = test.drop('id', axis=1).values

X = train.drop(['id', 'label'], axis=1).values
y = train['label'].values



params = {
#     'boosting_type': 'gbdt',
    'device': 'gpu',
    'objective': 'multiclass',
    'num_class': train['label'].nunique(),
    'metric': 'logloss',
#     'learning_rate': 0.1,
#     'num_leaves': 255,  
    'max_depth': -1,  
#     'min_child_samples': 100,  
#     'max_bin': 100,  
#     'subsample': 0.7,  
#     'subsample_freq': 1,  
#     'colsample_bytree': 0.7,  
#     'min_child_weight': 0,  
#     'subsample_for_bin': 200000,  
#     'min_split_gain': 0,  
#     'reg_alpha': 0,  
#     'reg_lambda': 0,  
   'nthread': 8,
    'verbosity': -1,
#     'scale_pos_weight':99 
    }

n_class = len(np.unique(y))

train_preds = np.zeros((len(X), n_class))
test_preds = np.zeros((len(test), n_class)) 
    # train['label'].nunique()))

cv_score = []
best_trees = []

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f'{i+1} fold')
    print(train_index)
    X_train_data = X[train_index] 
    X_valid_data = X[valid_index]

    print('slicing')
    y_train_data = y[train_index] 
    y_valid_data = y[valid_index]
    
    print(f'making dataset {i+1}')
    dtrain = lgb.Dataset(X_train_data, y_train_data)
    dvalid = lgb.Dataset(X_valid_data, y_valid_data, reference=dtrain)
    dtest = lgb.Dataset(test)

    bst = lgb.train(params, 
                     dtrain, 
                     num_boost_round=100, 
                     valid_sets=[dtrain, dvalid], 
                     verbose_eval=100,
                     early_stopping_rounds=10)
    
    best_trees.append(bst.best_iteration)
#     cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
    
#     cv_train[validate] += bst.predict(X_validate)
    
    preds = bst.predict_proba(dvalid)
    train_preds[valid_index] = preds
    test_preds += bst.predict_proba(test) / skf.n_splits
    
#     clf.fit(X_train_data, y_train_data)
    score = log_loss(y_valid_data, preds)
    print('{}-fold: logloss = {})'.format(i+1, score))
    
    cv_score.append(score)
#     train_preds[valid_index] = clf.predict_proba(X_valid_data)
#     test_preds += clf.predict_proba(test.values) / skf.n_splits

# print('\ntrain accuracy score = {:.3}'.format(accuracy_score(y, np.argmax(train_preds, axis=1))))
# print('train f1 score = {}'.format(f1_score(y, np.argmax(train_preds, axis=1), average=None)))


sub.iloc[:, 1:] = test_preds
sub.to_csv('submission.csv')
