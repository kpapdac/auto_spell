import unittest
from src import vocabulary

class test_vocabulary(unittest.TestCase):
    def setUp(self):
        self.text_label = [(0, 'this is a test.')]
        self.tokenize = 'basic_english'
    
    def test_get_vocab(self):
        voc = vocabulary.vocabulary()
        voc.tokenizer = self.tokenize
        voc.get_vocab(self.text_label)
        self.assertEqual(voc.vocab['this'], 5)
        self.assertEqual(voc.vocab['a'], 2)

    def test_set_text_pipeline(self):
        voc = vocabulary.vocabulary()
        voc.tokenizer = self.tokenize
        voc.get_vocab(self.text_label)
        voc.set_text_pipeline()
        self.assertEqual(voc.text_pipeline(self.text_label[0][1]), [5,3,2,4,1])
    
    def test_set_label_pipeline(self):
        voc = vocabulary.vocabulary()
        voc.tokenizer = self.tokenize
        voc.set_label_pipeline()
        self.assertEqual(voc.label_pipeline(self.text_label[0][0]), 0)
    
    def test_get_voc_size(self):
        voc = vocabulary.vocabulary()
        voc.tokenizer = self.tokenize
        voc.get_vocab(self.text_label)
        voc.get_voc_size()
        self.assertEqual(voc.vocab_size, 6)
    
    def test_get_num_class(self):
        voc = vocabulary.vocabulary()
        voc.tokenizer = self.tokenize
        voc.get_vocab(self.text_label)
        voc.get_num_class(self.text_label)
        self.assertEqual(voc.num_class, 1)


