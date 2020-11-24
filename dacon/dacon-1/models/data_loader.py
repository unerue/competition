import os
import re
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool 
from functools import partial
from tqdm import tqdm


def add_id_column(df, file_id):
    id_dataframe = pd.DataFrame({'id' : [file_id for i in range(df.shape[0])]})
    df = pd.concat([id_dataframe, df], axis=1)
    
    return df

def data_loader(path, train, nrows, **kwargs):
    '''
    Parameters:
    
    path: [str] train용 또는 test용 csv 파일들이 저장되어 있는 폴더 
    train: [boolean] train용 파일들 불러올 시 True, 아니면 False
    nrows: [int] csv 파일에서 불러올 상위 n개의 row 
    lookup_table: [pd.DataFrame] train_label.csv 파일을 저장한 변수 
    event_time: [int] 상태_B 발생 시간 
    normal: [int] 상태_A의 라벨
    
    Return:
    
    data: train 또는 test data
    '''
    # 1. 해당 파일 경로에서 확장자 제외한 파일 이름만 가지고 오기 
    # file_id = int(path.split('/')[-1].split('.')[0])
    regex = re.compile('[\d]+')
    file_id = int(regex.search(path)[0])
    
    # 2. train일 경우
    if train : 
        
        # 2-1 label 정보 저장
        lookup_table = kwargs['lookup_table']

        file_label = int(lookup_table[lookup_table['id'] == file_id]['label'])
        
        # 2-2 파일 읽기 
        data = pd.read_csv(path, nrows = nrows)
        
        # 2-3 id컬럼 추가 
        data = add_id_column(data, file_id)
        
        # 2-3 label컬럼 추가 
        event_time = kwargs['event_time']

        data['label'] = np.concatenate((np.repeat(kwargs['normal'], event_time),
                                        np.repeat(file_label, data.shape[0]-event_time)))

    # 3. test일 경우 
    else : 
        
        # 3-1 파일 읽기
        data = pd.read_csv(path, nrows = nrows)
        
        # 3-2 id컬럼 추가 
        data = add_id_column(data, file_id)
        
    return data


def data_loader_all(func, path, train, nrows, **kwargs):
    '''
    Parameters:
    
    func: 하나의 csv파일을 읽는 함수 
    path: [str] train용 또는 test용 csv 파일들이 저장되어 있는 폴더 
    train: [boolean] train용 파일들 불러올 시 True, 아니면 False
    nrows: [int] csv 파일에서 불러올 상위 n개의 row 
    lookup_table: [pd.DataFrame] train_label.csv 파일을 저장한 변수 
    event_time: [int] 상태_B 발생 시간 
    normal: [int] 상태_A의 라벨
    
    Return:
    
    combined_df: 병합된 train 또는 test data
    '''
    
    # 읽어올 파일들만 경로 저장 해놓기 
    files_in_dir = os.listdir(path)[:100]
    
    files_path = [path+'/'+file for file in files_in_dir]
    
    if train :
        func_fixed = partial(func, nrows = nrows, train = True, lookup_table = kwargs['lookup_table'], event_time = kwargs['event_time'], normal = kwargs['normal'])
        
    else : 
        func_fixed = partial(func, nrows = nrows, train = False)
    
    
    # 여러개의 코어를 활용하여 데이터 읽기 
    # if __name__ == '__main__':
    #    pool = Pool(processes = multiprocessing.cpu_count()) 
    #    df_list = list(tqdm(pool.imap(func_fixed, files_path), total = len(files_path)))
    #    pool.close()
    #    pool.join()
        
    pool = Pool(processes = multiprocessing.cpu_count()) 
    df_list = list(tqdm(pool.imap(func_fixed, files_path), total = len(files_path)))
    pool.close()
    pool.join()
    # 데이터 병합하기 
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df