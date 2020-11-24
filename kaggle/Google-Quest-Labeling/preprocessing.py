# Import Module
import os
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import sentencepiece as spm

from tqdm import tqdm
from glob import glob
from gensim.models.word2vec import Word2Vec

# Import customize decorator
from decorator import elapsed_time

@elapsed_time
def main(args):
    # Read data
    logger = logging.getLogger('Preprosser')
    logger.info('Load and preprocessing dataset....')
    # print('Load & process data...')
    # start_time = time.time()

    train_dat_total = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test_dat_total = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    # total 30 labels
    labels = list(train_dat_total.columns)[11:]

    # Train / Valid split
    data_len = len(train_dat_total)
    train_len = int(data_len * 0.8)

    train_index = np.random.choice(data_len, train_len, replace = False) 
    valid_index = list(set(range(data_len)) - set(train_index))

    train_dat = train_dat_total.iloc[train_index]
    train_dat.index = range(len(train_dat))

    valid_dat = train_dat_total.iloc[valid_index]
    valid_dat.index = range(len(valid_dat))

    # Text data extract
    title_list = train_dat['question_title'].tolist()
    body_list = train_dat['question_body'].tolist()
    ans_list = train_dat['answer'].tolist()
    total_question = title_list + body_list + ans_list

    # print('{:.4f}min spend..'.format((time.time() - start_time) / 60))

    # Byte Pair Encoding (BPE) fitting
    logger.info('BPE fitting...')
    # print('BPE fitting...')
    # start_time = time.time()

    if not os.path.exists('./preprocessing/m_{}.model'.format(args.vocab_size)):
        # Make hanja text to train vocab
        with open('./preprocessing/text.txt', 'w') as f:
            for text in total_question:
                f.write(f'{text}\n')
        
        spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(
            f'--input=./preprocessing/text.txt --model_prefix=./preprocessing/m_{args.vocab_size} '
            f'--vocab_size={args.vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
            f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
            f'--model_type={args.parsing_type} --user_defined_symbols=[SEP]')

    # Vocabulary setting
    vocab_list = list()
    with open('./preprocessing/m_{}.vocab'.format(args.vocab_size)) as f:
        for line in f:
            vocab_list.append(line[:-1].split('\t')[0])

    word2id = {w: i for i, w in enumerate(vocab_list)}

    # Encode text data
    spm_ = spm.SentencePieceProcessor()
    spm_.Load("./preprocessing/m_{}.model".format(args.vocab_size))

    train_dict = {
        'title_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in title_list],
        'body_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in body_list],
        'ans_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in ans_list]
    }

    valid_dict = {
        'title_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in valid_dat['question_title']],
        'body_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in valid_dat['question_body']],
        'ans_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in valid_dat['answer']]
    }

    test_dict = {
        'title_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in test_dat_total['question_title']],
        'body_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in test_dat_total['question_body']],
        'ans_encode': [[args.bos_id] + spm_.EncodeAsIds(text) + [args.eos_id] for text in test_dat_total['answer']]
    }

    # Label setting
    for col in labels:
        train_dict[col] = train_dat[col]
        valid_dict[col] = valid_dat[col]

    print('{:.4f}min spend..'.format((time.time() - start_time) / 60))

    print('Save data...')
    start_time = time.time()

    with open(f'./preprocessing/preprocessing_{args.vocab_size}.pkl', 'wb') as f:
        pickle.dump({
            'train': train_dict,
            'valid': valid_dict,
            'test': test_dict
        }, f)

    print('{:.4f}min spend..'.format((time.time() - start_time) / 60))

    # Word2Vec
    # print('Word2Vec Training...')
    logger.info('Training Word2Vec...')
    start_time = time.time()

    # Korean Pre-processing
    str_indices = list()
    for ind in train_dict['title_encode'] + train_dict['body_encode'] + train_dict['ans_encode']:
        ind_list = list()
        for ix in ind:
            ind_list.append(str(ix))
        str_indices.append(ind_list)

    # Word2Vec training
    word2vec_hj = Word2Vec(str_indices, window=3, size=args.embedding_dim, min_count=1, workers=8, sg=1)
    word2vec_hj.save(f'./preprocessing/word2vec_{args.vocab_size}_{args.embedding_dim}.model')
    print('{:.4f}min spend..'.format((time.time() - start_time) / 60))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing')
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Vocabulary Size; Default is 32000')
    # parser.add_argument('--data_path', default='./data', type=str, help='data path')
    parser.add_argument('--data_path', default='input', type=str, help='data path')
    parser.add_argument('--parsing_type', default='bpe', type=str, help='parsing_type of sentencepiece model')
    parser.add_argument('--pad_id', default=0, type=int, help='pad index')
    parser.add_argument('--bos_id', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_id', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_id', default=3, type=int, help='index of unk token')
    parser.add_argument('--embedding_dim', default=256, type=int, help='embedding dimension')
    args = parser.parse_args()

    main(args)
