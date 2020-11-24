import logging
import os
import time
import pickle
import argparse
import torch
from gensim.models import Word2Vec

# Import Custom  Modules
from dataset import CustomDataset, PadCollate
from torch.dataset import DataLoader

def main(args):
    # Data Load
    with open(args.data_path, 'rb') as f:
        data  = pickle.load(f)

    # Dataset setting
    dataset_dict = {
        'train': 
            CustomDataset(data['train']['title_encode'], 
            data['train']['body_encode'], 
            data['train']['ans_encode'], 
            min_len=args.min_len, 
            max_len=args.max_len),
        'valid':
            CustomDataset(data['valid']['title_encode'], 
            data['valid']['body_encode'], data['valid']['ans_encode'], 
            min_len=args.min_len, 
            max_len=args.max_len),
        'test': 
            CustomDataset(data['test']['title_encode'], 
            data['test']['body_encode'], 
            data['test']['ans_encode'], 
            min_len=args.min_len, 
            max_len=args.max_len)
    }

    dataloader_dict = {
        'train': 
            DataLoader(dataset_dict['train'], 
            collate_fn=PadCollate(), 
            drop_last=True, 
            pin_memory=True, 
            batch_size=args.batch_size),
        'valid':  
            DataLoader(dataset_dict['valid'], 
            collate_fn=PadCollate(), 
            drop_last=True, 
            pin_memory=True, 
            batch_size=args.batch_size),
        'test':  
            DataLoader(dataset_dict['test'], 
            collate_fn=PadCollate(), 
            drop_last=True, 
            pin_memory=True, 
            batch_size=args.batch_size)
    }

    # Word2Vec initialization
    word2vec = Word2Vec.load(args.embedding_path)

    # src_word2vec.init_sims(replace=True)
    # for word, id_ in src_word2id.items():
    #     try:
    #         model.src_embedding.token.weight.data[id_] = torch.FloatTensor(src_word2vec.wv[word])
    #     except:
    #         continue

if __name__ == "__main__":
    logging.basicConfig(format=logging.INFO)
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--data_path', default='./preprocessing/preprocessing_32000.pkl', type=str, help='path of pickle data path')
    parser.add_argument('--embedding_path', default='./preprocessing/word2vec_32000_256.model', type=str, help='path of trained embedding vectors')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    parser.add_argument('--min_len', default=4, type=int, help='minimum length of the sentence')
    parser.add_argument('--max_len', default=300, type=int, help='maximum length of the sentence')
    args = parser.parse_args()
    # ?
    main(args)
