import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys

from src.models.base_model import Encoder, Decoder

class StepDecoder(nn.modu):
    def __init__(self, embed_sz, hidden_sz, vocab_sz,
                 Embedding=None, Classifier=None):
        super(StepDecoder, self).__init__()

        ''' decoder states '''
        self.embed_sz    = embed_sz
        self.hidden_sz   = hidden_sz
        self.vocab_sz    = vocab_sz

        ''' decoder layers'''
        self.embed       = None
        self.rnn         = nn.LSTMCell()
        self._hW         = None
        self._cW         = None
        self.classifier  = None
        if Embedding is None:
            self.embed = nn.Embedding(vocab_sz, hidden_sz)
        else:
            self.embed = Embedding(vocab_sz, hidden_sz)
        
        if Classifier is None:
            self.classifier = nn.Linear(self.hidden_sz, self.vocab_sz)
        else: 
            self.classifier = Classifier(self.hidden_sz self.vocab_sz)
        
    def init_weights(self):
        self._hW = torch.uniform(-1, 1)
        self._cW = torch.uniform(-1, 1)
    

    def forward(self, args):
        pass

    def sample(self, args):


class StepEncoderDecoder(nn.Module):
    def __init__(self, vocab_sz, embed_sz, hidden_sz,
                Embedding=None, Classifier=None):
        pass
    
    def forward(self, images, manager):
        pass
    
    def parameters(self):
        return list()

