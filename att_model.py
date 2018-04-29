import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from OldModel import *;
import sys;

RESNET_SHAPE = (2048, 7, 7)
RESNET_LAYER = -2
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:RESNET_LAYER]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # self.linear = nn.Linear(resnet.fc.in_features*7*7, embed_size)
        self.LD = reduce(lambda x,y:x*y, RESNET_SHAPE);
        self.L = RESNET_SHAPE[1]*RESNET_SHAPE[2];
        self.D = RESNET_SHAPE[0];
        self.linear = nn.Linear(self.D, self.D)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.bn2D = nn.BatchNorm2d(self.D, momentum=0.01)
        self.init_weights();

    def init_weights(self):
        """Initialize the weights."""
        # self.linear.weight.data.normal_(0.0, 0.02)
        torch.nn.init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.fill_(0)

    #(N,L,D) --> (N,L,D)
    def _project_features(self,features):
        # w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
        features_flat = features.view(-1, self.D);
        # features_flat = tf.reshape(features, [-1, self.D])
        # features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
        features_proj = self.linear(features_flat);
        features_proj = features.view(-1, self.L, self.D);
        return features_proj


    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data) #(N, D, L', L') where L' = sqrt(L)
        features = self.bn2D(features); # (N, D, L', L')

        att_features = features.view((features.shape[0], RESNET_SHAPE[0], \
                                      RESNET_SHAPE[1]*RESNET_SHAPE[2]));
        att_features = att_features.permute(0,2,1); # (N,L,D)

        features_proj = self._project_features(att_features);
        print 'features_proj', features_proj.shape;
        sys.exit();
        features = features.view(features.size(0), -1)
        print 'features2', features.shape
        proj_features = self.bn(self.linear(features))
        print 'features.shape', proj_features.shape
        sys.exit();
        return proj_features, att_features;


class DecoderRNN(nn.Module):
    def __init__(self, embed_size,hidden_size, vocab_size, num_layers ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.V = vocab_size
        self.M = embed_size
        self.H =  hidden_size
        self.L = RESNET_SHAPE[1]*RESNET_SHAPE[2]; #(7 x 7)
        self.D = RESNET_SHAPE[0]; #(2048)
        print 'embed_size(M): ',embed_size, 'hidden_size(H): ',hidden_size, \
        'vocab_size(V): ',vocab_size, 'L: ', self.L, 'D: ', self.D, 'num_layers: ',num_layers
        self.embed = nn.Embedding(self.V, self.M)
        self.lstm = nn.LSTM(self.M, self.H, num_layers, batch_first=True)
        self.linear = nn.Linear(self.H, self.V)
        self.affine_lstm_init = nn.Linear(self.M, self.H);
        self.lstm_cell = nn.LSTMCell(self.M, self.H);
        self.init_weights();
        self.hidden_size = hidden_size;
        self.embed_size = embed_size;
        ''' in forward>
        T = n_time_step
        N = batch size
        '''
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.affine_lstm_init.weight.data.uniform_(-0.1, 0.1)
        self.affine_lstm_init.bias.data.fill_(0)

    def forward(self, projected_features, features, captions, lengths):
        """Decode image feature vectors and generates captions.
            projected_features.shape = (N, L*D), i.e (5, 49*2048 = 100352)
            features = (N, L, D) i.e (N, 49, 2048)
        """
        N, T = captions.shape;
        embeddings = self.embed(captions) # = (N, M)
        next_c = Variable(torch.zeros(N, self.hidden_size))#.cuda() #need cuda
        next_h = self.affine_lstm_init(projected_features); # (N,L,D) * ()
        h_list = []
        print ("OK!!");
        sys.exit()
        for i in range(0,T):
            # expects input = (N, M), h,c = (N, H)
            next_h, next_c = self.lstm_cell(embeddings[:,i,:], (next_h, next_c));
            h_list.append(next_h);
        hiddens = torch.cat(h_list);
        outputs = self.linear(hiddens)
        return outputs;

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()
