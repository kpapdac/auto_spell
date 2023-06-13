import unittest
import torch
from src import model, vocabulary

class test_model(unittest.TestCase):
    def setUp(self):
        self.text_label = [(0, 'this is a test.')]
        self.voc = vocabulary.vocabulary()
        self.voc.tokenizer = 'basic_english'
        self.voc.get_vocab(self.text_label)
        self.voc.get_voc_size()
        self.voc.get_num_class(self.text_label)
        self.num_embeddings = self.voc.vocab_size
        self.embed_dim = 2
        self.num_class = self.voc.num_class

    def test_textclassification(self):
        model_ = model.TextClassificationModel(self.num_embeddings, self.embed_dim, self.num_class)
        self.assertEqual([i[0] for i in model_.named_parameters()], ['embedding.weight', 'fc.weight', 'fc.bias'])
        self.assertEqual([i[1] for i in model_.named_parameters() if i[0]=='embedding.weight'][0].shape, torch.Size([6,2]))
        self.assertEqual([i[1] for i in model_.named_parameters() if i[0]=='fc.weight'][0].shape, torch.Size([1,2]))
        self.assertEqual([i[1] for i in model_.named_parameters() if i[0]=='fc.bias'][0].shape, torch.Size([1]))
    
    def test_LSTM(self):
        n_hidden = 2
        n_rnn = 2
        model_ = model.LSTM(self.num_embeddings, self.embed_dim, n_hidden, n_rnn, self.num_class)
        self.assertEqual(len([i[0] for i in model_.named_parameters()]), 11)
