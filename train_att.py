import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from data_loader import get_loader_coco
from build_vocab import Vocabulary
from att_model import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from masked_cross_entropy import compute_loss;
import json;
import sys;
import time;
#
NAME = 'DEBUG'
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def name_pretrained_sizes(nl, emb, hid, xoder, epoch, i): return '%d_%d_%d_%s-%d-%d.pkl' %(nl, emb, hid, xoder, epoch+1, i+1)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    print '----- DATA_LOADER -- loaded data ----'
    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, batch_size=args.batch_size)

    if torch.cuda.is_available():
        print '---- USING GPU ---- '
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)
    t0 = time.time();
    print ''
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            N, T = captions.shape;
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            targets2 = captions.view((captions.shape[0]*captions.shape[1], ));
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            projected_features, features = encoder(images)
            outputs = decoder(projected_features, features, captions, lengths)
            lengths = torch.cuda.LongTensor(lengths);
            # lengths = torch.LongTensor(lengths);
            # loss = criterion(outputs, targets);
            loss = compute_loss(outputs, captions, lengths) #targets
            loss.backward()
            # for name, param in decoder.named_parameters():
            #     if param.requires_grad and name == 'h0':
            #         print name, param.data
            #     if param.requires_grad and name == 'c0':
            #         print name, param.data
            optimizer.step()
            # Print log info
            '''print
            projected_features, features = encoder(images[0].unsqueeze(0));
            sampled_ids = decoder.sample(projected_features, features);
            sampled_caption = []
            sampled_ids = sampled_ids.cpu().data.numpy()
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            print sentence
            sys.exit()
            '''

            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Time: %.4f'
                      %(epoch, args.num_epochs, i, total_step,
                        loss.data[0], np.exp(loss.data[0]), time.time()-t0))
            # else:
            #     print '%s - loss: %.4f - time: %.4f'%(i, loss.data[0], time.time()-t0);

            # Save the models
            # if (i+1) % args.save_step == 0:
                #pass

    print 'saving final model';
    print '%s - loss: %.4f - time: %.4f'%(i, loss.data[0], time.time()-t0);
    torch.save(decoder.state_dict(),
               os.path.join(args.model_path,
                            # name_pretrained_sizes(args.num_layers, args.embed_size, args.hidden_size, "decoder", epoch, i)))
                            '%s-att-decoder-%d-%d.pkl' %(NAME, epoch+1, i+1)))
    torch.save(encoder.state_dict(),
               os.path.join(args.model_path,
                            # name_pretrained_sizes(args.num_layers, args.embed_size, args.hidden_size, "encoder", epoch, i)))
                            '%s-att-encoder-%d-%d.pkl' %(NAME,epoch+1, i+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/ATT' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./coco/annotations/sm_captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=50,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=32 , #256
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=64 , #512
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5) #128
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    args = parser.parse_args()
    print(args)
    main(args)
