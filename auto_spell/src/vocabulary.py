from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

class vocabulary:
    def __init__(self):
        self.__name = ''
        self.vocab = None
        self.text_pipeline = None
        self.label_pipeline = None
        self.vocab_size = None
        self.num_class = None
    
    @property
    def tokenizer(self):
        return self.__name
    
    @tokenizer.setter
    def tokenizer(self, val):
        self.__name = val

    def tokenizer_(self):
        return get_tokenizer(self.__name)
    
    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer_()(text)

    def get_vocab(self, data):
        vocab = build_vocab_from_iterator(self.yield_tokens(data), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab

    def set_text_pipeline(self):
        self.text_pipeline = lambda x: self.vocab(self.tokenizer_()(x))

    def set_label_pipeline(self):
        self.label_pipeline = lambda x: int(x)

    def get_voc_size(self):
        self.vocab_size = len(self.vocab)

    def get_num_class(self, data_iter):
        self.num_class = len(set([label for (label, text) in data_iter]))