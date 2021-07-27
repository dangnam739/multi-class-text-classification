from utils.data_utils import *
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, df, words_list, word_vectors, max_sen_len):
        self.words_list = words_list
        self.word_vectors = word_vectors
        self.df = df
        self.sentences = self.df["text"]
        self.labels = self.df["class_id"]
        self.max_sen_len = max_sen_len

    def __getitem__(self, index):
        sentence = self.sentences[index]
        txt = text2ids(sentence, self.max_sen_len, self.words_list)
        label = torch.LongTensor([self.labels[index]])

        return txt, label

    def __len__(self):
        return len(self.df)
