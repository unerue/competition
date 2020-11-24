# Import Module
import torch
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, title_list, body_list, ans_list, min_len=4, max_len=300):
        title_list, body_list, ans_list = zip(*[(t, b, a) for t, b, a in zip(title_list, body_list, ans_list)\
            if min_len <= len(t) <= max_len and min_len <= len(b) <= max_len and min_len <= len(a) <= max_len])

        self.text_data = [(t, b, a) for t, b, a in zip(title_list, body_list, ans_list)]
        self.num_data = len(self.text_data)

    def __getitem__(self, index):
        title, body, answer = self.text_data[index]        
        return title, body, answer
    
    def __len__(self):
        return self.num_data


class PadCollate:
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences

        sentences_list = zip(*batch)
        sentences_list = [pack_sentence(sentences) for sentences in sentences_list]        
        return tuple(sentences_list)

    def __call__(self, batch):
        return self.pad_collate(batch)