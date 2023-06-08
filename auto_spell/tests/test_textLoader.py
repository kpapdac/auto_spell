from datasets import load_dataset
import unittest
from src import textLoader
from torch.utils.data import DataLoader

class test_preparetextloader(unittest.TestCase):
    def setUp(self):
        self.raw_dataset = load_dataset('sms_spam')
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.raw_dataset['train'], 'sms', 'label')
        self.data = prep.convert_to_text_label()
        self.train_data, self.test_data = prep.get_train_test_split(self.data)

    def test_prepare_textloader_logistic(self):
        prep = textLoader.prepareTextLabelLoaderLogisticNN(self.raw_dataset['train'], 'sms', 'label')
        self.train_dataloader = DataLoader(self.train_data, batch_size=8, shuffle=False, collate_fn=prep.collate)
        self.test_dataloader = DataLoader(self.test_data, batch_size=8, shuffle=False, collate_fn=prep.collate)
        self.assertLess(len(self.test_data), len(self.train_data))
        self.assertLess(len(self.test_dataloader.dataset), len(self.train_dataloader.dataset))

    def test_prepare_textloader_rnn(self):
        prep = textLoader.prepareTextLabelLoaderRNN(self.raw_dataset['train'], 'sms', 'label')
        self.train_dataloader = DataLoader(self.train_data, batch_size=8, shuffle=False, collate_fn=prep.collate)
        self.test_dataloader = DataLoader(self.test_data, batch_size=8, shuffle=False, collate_fn=prep.collate)
        self.assertLess(len(self.test_data), len(self.train_data))
        self.assertLess(len(self.test_dataloader.dataset), len(self.train_dataloader.dataset))
