# Imports
import abc
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from datetime import datetime

import random
import collections

import torch
import torchtext

from torchtext.data.functional import to_map_style_dataset
from torchtext.transforms import Sequential, ToTensor
from torchtext.data import get_tokenizer

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class prepareTextLabelLoader:
    def __init__(self, data, text_name: str, label_name: str):
        self.data = data
        self.text_name = text_name
        self.label_name = label_name

    #@property
    def convert_to_text_label(self):
        return [(self.data[self.label_name][i], self.data[self.text_name][i]) \
             for i in range(len(self.data[self.text_name]))]
    
    @staticmethod
    def get_train_test_split(data, split_ratio = 0.7):
        random.shuffle(data)
        n_train = int(len(data) * split_ratio)
        train_data = data[:n_train]
        test_data = data[n_train:]
        return train_data, test_data
    
    #@abc.abstractmethod
    def collate(self):
        pass

class prepareTextLabelLoaderLogisticNN(prepareTextLabelLoader):
    def __init__(self, data, text_name: str, label_name: str, text_pipeline, label_pipeline):
        super().__init__(data, text_name, label_name)
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def collate(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

class prepareTextLabelLoaderRNN(prepareTextLabelLoader):
    def __init__(self, data, text_name: str, label_name: str, pad_idx: int, text_pipeline, label_pipeline):
        super().__init__(data, text_name, label_name)
        self.pad_idx = pad_idx
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def collate(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        labels = label_list
        text = nn.utils.rnn.pad_sequence(text_list, padding_value=self.pad_idx, batch_first=True)
        return labels, text
