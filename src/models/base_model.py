import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys;

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    # import sys
    def sample(self, features, manager, sample_size = 15):
        sampled_token_idxs = []
        ctx = features.unsqueeze(1)
        next_token_id = -1
        token_ids = []
        state = None
        for i in range(sample_size):
            if next_token_id == manager.wtoi["<end>"]:
                break
            _, (h_next, c_next) = self.lstm(ctx, state) # h_next == output, h_next = (1, 1, hidden_size)
            score_i = self.linear(h_next.squeeze(0))
            next_token_id = torch.argmax(score_i, 1)
            # print('next_token_id', next_token_id.shape)
            ctx = self.embed(next_token_id)
            # print(ctx.shape)
            ctx = ctx.unsqueeze(0)
            state = (h_next, c_next)
            # logits = self.linear(outputs.)
            token_ids.append(next_token_id)
        token_ids = torch.cat(token_ids, 0)
        return token_ids

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size = 64, hidden_size = 128,  
                 num_layers = 1):

        super(EncoderDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions, lengths):
        image_features = self.encoder(images)
        out = self.decoder(image_features, captions, lengths)
        return out

    def sample(self, images, manager):
        image_features = self.encoder(images)
        sampled_token_idxs = self.decoder.sample(image_features, manager)
        return sampled_token_idxs

    def parameters(self):
        return list(self.encoder.bn.parameters())\
                    +list(self.encoder.linear.parameters())\
                    +list(self.decoder.parameters())
