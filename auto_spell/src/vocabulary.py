class vocabulary:
    def __init__(self):
        self.__name = ''
    
    @property
    def tokenizer(self):
        return self.__name
    
    @tokenizer.setter
    def tokenizer(self, val):
        self.__name = val