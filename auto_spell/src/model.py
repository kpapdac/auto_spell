from torch import nn
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationModel(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return F.log_softmax(self.fc(embedded), dim=1)

class TextClassificationModel2D(nn.Module):

    def __init__(self, vocab_size, embed_dim, H, num_class):
        super(TextClassificationModel2D, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, num_class)
        )
        self.init_weights()
        
    def init_weights(self):
        initrange = 5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
            
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class LSTM(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_outputs):
        super(LSTM, self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers
        self.embed = nn.Embedding(self.V, self.D, sparse=False)
        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=False)
        self.fc = nn.Linear(self.M, self.K)
    
    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(1), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(1), self.M).to(device)


        out = self.embed(X)
        # get RNN unit output
        out, _ = self.rnn(out, (h0, c0))

        # max pool
        out, _ = torch.max(out, 1)

        # we only want h(T) at the final time step
        out = self.fc(out)
        return out