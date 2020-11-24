import gc
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold


class BG:
    black = '\033[40m'
    red = '\033[41m'
    green = '\033[42m'
    end = '\033[1;m'
    
def trans_time(df, start=0, end=4319):
    results = {}
    minutes = pd.Series((df.loc[start:end, 'id'] % 144).astype(int), name='Minutes')
    hours = pd.Series((df.iloc[start:end].index % 144 / 6).astype(int), name='Hours')
    results['minutes'] = minutes
    results['hours'] = hours
    
    min_in_day = 24 * 6
    hour_in_day = 24
    results['min_in_day'] = min_in_day
    results['hour_in_day'] = hour_in_day

    minute_sin = pd.Series(np.sin(np.pi*minutes / min_in_day), name='MinSin')
    minute_cos = pd.Series(np.cos(np.pi*minutes / min_in_day), name='MinCos')
    results['minute_sin'] = minute_sin
    results['minute_cos'] = minute_cos

    hour_sin  = pd.Series(np.sin(np.pi*hours / min_in_day), name='HourSin')
    hour_cos  = pd.Series(np.cos(np.pi*hours / min_in_day), name='HourCos')
    results['hour_sin'] = hour_sin
    results['hour_cos'] = hour_cos
    
    return results

def same_min_max(df):
    columns = df.columns[df.max() == df.min()]
    return df.drop(df.columns[df.max() != df.min()], axis=1)

def custom_mse(preds, train_data):
    labels = train_data.get_label().astype('float')
    diff = abs(labels - preds)
    less_then_one = np.where(diff < 1, 0, diff)
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    
    return 'mse', score, False

def main():
    parser = argparse.ArgumentParser(description='Test Submission')
    parser.add_argument('--column', type=str)
    parser.add_argument('--filename', type=str, default='submission')
    args = parser.parse_args()
    
    print(f'{BG.red}Starting based on the "{args.column}"...{BG.end}\n')
    
    # .csv 파일 불러오기
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    sub = pd.read_csv('input/sample_submission.csv')

    # 서브셋
    subset_train = train.reset_index(drop=True).loc[:4319, 'id':'X39'].copy()
    subset_target = train.reset_index(drop=True).loc[:4319, args.column].copy()
    
    # 시간변수
    times = trans_time(subset_train)
    concated_train = pd.concat([
        subset_train, 
        times['minutes'], 
        times['hours'], 
        times['minute_sin'], 
        times['minute_cos'],
        times['hour_sin'],
        times['hour_cos'],
        subset_target,
    ], ignore_index=True, axis=1)
    
    # 변수명 변경
    columns = ['id'] + [f'X{i:02}' for i in range(40)] + [
        'Minutes', 'Hours', 'MinSin', 'MinCos', 'HourSin', 'HourCos', 'Target'
    ]    
    concated_train.columns = columns

    # 훈련 데이터셋
    X = concated_train.drop(['Target'], axis=1).copy()
    y = concated_train['Target'].copy()
    
    # 불필요한 변수 삭제
    X = X.drop(same_min_max(X), axis=1)
    
    # 테스트셋 만들기
    times_test = trans_time(test, end=11520)
    concated_test = pd.concat([
        test, 
        times_test['minutes'], 
        times_test['hours'], 
        times_test['minute_sin'], 
        times_test['minute_cos'],
        times_test['hour_sin'],
        times_test['hour_cos'],
    ], ignore_index=True, axis=1)

    test_data = concated_test.copy()
    columns = ['id'] + [f'X{i:02}' for i in range(40)] + [
        'Minutes', 'Hours', 'MinSin', 'MinCos', 'HourSin', 'HourCos'
    ]
    test_data.columns = columns
    
    # 불필요한 변수 삭제
    test = test_data.drop(['X14', 'X16', 'X19'], axis=1).copy()
    X = X.drop(['id', 'X03', 'X10', 'X13', 'X21', 'X24', 'X36', 'X39'], axis=1)
    test = test.drop(['id', 'X03', 'X10', 'X13', 'X21','X24', 'X36', 'X39'], axis=1)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'learning_rate': 0.01,
        'max_depth': -1,  
        'bagging_fraction': 0.5,
        'feature_fraction': 0.5,
        'nthread': -1,
        'verbosity': -1,
    }

    # OOF 전략
    train_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    feature_importances = []
    kfold = KFold(n_splits=5, shuffle=True)
    print(f'Starting training using out of folds...\n')
    for i, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        X_train_data = X.iloc[train_index] 
        X_valid_data = X.iloc[valid_index]
        
        y_train_data = y.iloc[train_index] 
        y_valid_data = y.iloc[valid_index]

        dtrain = lgb.Dataset(X_train_data, y_train_data)
        dvalid = lgb.Dataset(X_valid_data, y_valid_data)
        dtest = lgb.Dataset(test)

        bst = lgb.train(params, 
                        dtrain,
                        feval=custom_mse,
                        num_boost_round=2000, 
                        valid_sets=[dtrain, dvalid], 
                        verbose_eval=None, 
                        early_stopping_rounds=1000)

        feature_importances.append(bst.feature_importance())
        preds_by_train = bst.predict(X_train_data, num_iteration=bst.best_iteration)
        preds_by_valid = bst.predict(X_valid_data, num_iteration=bst.best_iteration)
        train_preds[valid_index] = preds_by_valid
        
        # 테스트셋 예측하기
        test_preds += bst.predict(test.values, num_iteration=bst.best_iteration) / kfold.n_splits
        
        train_score = custom_mse(preds_by_train, dtrain)[1]
        valid_score = custom_mse(preds_by_valid, dvalid)[1]
        print(f'{i+1}-fold: train mse = {train_score:.4f}\t valid mse = {valid_score:.4f}')

    print(f'\n{BG.green}Saving "{args.filename}.csv"...{BG.end}')
    sub.loc[:, 'Y18'] = test_preds
    sub.to_csv(f'{args.filename}.csv', index=False)
    print(f'Done!')
    
if __name__ == '__main__':
    main()


