import unittest
import torch
from src import model, optimize, textLoader, vocabulary
from torch.utils.data import DataLoader

class test_optimize(unittest.TestCase):
    def setUp(self):
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
        self.model = model.TextClassificationModel(self.num_embeddings, self.embed_dim, self.num_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        LR = 5
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR)

    def test_train(self):
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.text_label, 'sms', 'label', self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model, self.criterion, self.optimizer)        
        opt.train()
        opt.evaluate()

    def test_hypertune(self):
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.text_label, 'sms', 'label', self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model, self.criterion, self.optimizer)        
        opt.hypertuning(epochs=10)

    def test_predict(self):
        text = 'this is a test.'
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.text_label, 'sms', 'label', self.voc.text_pipeline, self.voc.label_pipeline)
        self.train_dataloader = DataLoader(self.text_label, batch_size=8, shuffle=False, collate_fn=prep.collate)
        opt = optimize.optimize(self.train_dataloader, self.train_dataloader, self.model, self.criterion, self.optimizer)        
        opt.hypertuning(epochs=10)
        opt.predict(text, self.voc.text_pipeline)
