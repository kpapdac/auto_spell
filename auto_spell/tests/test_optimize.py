import unittest
import torch
from src import model, optimize, textLoader, vocabulary
from torch.utils.data import DataLoader
from datasets import load_dataset

class test_optimize(unittest.TestCase):
    def setUp(self):
        # prep = textLoader.prepareTextLabelLoader(load_dataset('sms_spam')['train'], 'sms', 'label')
        # self.text_label = prep.convert_to_text_label()
        self.text_label = [(0, 'this is a test.'), (1, 'this is another test.')]
        self.voc = vocabulary.vocabulary()
        self.voc.tokenizer = 'basic_english'
        self.voc.get_vocab(self.text_label)
        self.voc.set_text_pipeline()
        self.voc.set_label_pipeline()
        self.voc.get_voc_size()
        self.voc.get_num_class(self.text_label)
        self.num_embeddings = self.voc.vocab_size
        self.embed_dim = 2
        self.num_class = self.voc.num_class
        self.model_rnn = model.LSTM(self.num_embeddings, self.embed_dim, 15, 1, self.num_class)
        self.model = model.TextClassificationModel(self.num_embeddings, self.embed_dim, self.num_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        LR = 5
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR)

    def test_train(self):
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.text_label, 'sms', 'label', self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model, self.criterion, self.optimizer, False)        
        opt.train(epoch=1)
        opt.evaluate()

    def test_hypertune(self):
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.text_label, 'sms', 'label', self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model, self.criterion, self.optimizer, False)        
        opt.hypertuning(epochs=10)

    def test_predict(self):
        text = 'this is a test.'
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.text_label, 'sms', 'label', self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model, self.criterion, self.optimizer, False)        
        opt.hypertuning(epochs=10)
        opt.predict(text, self.voc.text_pipeline)

    def test_train_rnn(self):
        pad_token = '<PAD>'
        prep = textLoader.prepareTextLabelLoaderRNN(self.text_label, 'sms', 'label', self.voc.vocab[pad_token], self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model_rnn, self.criterion, self.optimizer, True)        
        opt.train(epoch=1)
        opt.evaluate()

    def test_hypertune_rnn(self):
        pad_token = '<PAD>'
        prep = textLoader.prepareTextLabelLoaderRNN(self.text_label, 'sms', 'label', self.voc.vocab[pad_token], self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model_rnn, self.criterion, self.optimizer, True)        
        opt.hypertuning(epochs=10)

    # def test_predict_rnn(self):
    #     text = 'this is a test.'
    #     pad_token = '<PAD>'
    #     print(torch.tensor(self.voc.text_pipeline(text)).size)
    #     # errors in model X.size(1) as it is a single text. test not appropriate, to review.
    #     prep = textLoader.prepareTextLabelLoaderRNN(self.text_label, 'sms', 'label', self.voc.vocab[pad_token], self.voc.text_pipeline, self.voc.label_pipeline)
    #     self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
    #     opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model_rnn, self.criterion, self.optimizer, True)        
    #     opt.hypertuning(epochs=10)
    #     opt.predict(text, self.voc.text_pipeline)
