import sys, os
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TYPES = ['loss', 'f1', 'precision','recall']
SPLIT_TYPES = ['train', 'test', 'val']
SPLIT_TO_COLOR = {
    'train': 'r',
    'test': 'g',
    'val': 'b'
}

def plot_eval(data, split_type, name="loss"):
    if split_type not in SPLIT_TYPES:
        raise ValueError("No such split_type `{}`".format(split_type))    
    
    if name not in data:
        raise KeyError("no data of key-type = {}".format(name))
    
    Y = data[name][split_type]
    X = range(len(Y))
    color_format = '{}-'.format(SPLIT_TO_COLOR[split_type])
    plt.plot(X, Y, color_format)
    plt.ylabel(name)
    plt.xlabel('epochs')
    plt.title("{} for {} set".format(name, split_type))
    plt.show()

def run_model(model, data_loader, criterion, optimizer, train=False, epoch=0, 
              log_step=2, device=device):
    if train:
        model.train()
    else:
        model.eval()
    
    losses = []
    logged_losses = []
    for i, batch in enumerate(data_loader):
        _ , images, captions, caption_lengths = batch
        optimizer.zero_grad()

        images = images.to(device)
        captions = captions.to(device)

        packed_captions = pack_padded_sequence(captions, caption_lengths, batch_first=True)
        targets = packed_captions.data

        logits = model(images, captions, caption_lengths)
        loss = criterion(logits, targets)
        # print('predicted', logits.max(1))
        # print('targets', targets)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if (i+1)%log_step == 0:
            loss_mean = np.mean(losses)
            print("ep={} step={} loss={}".format(epoch, i+1, loss_mean))
            logged_losses.append(loss_mean)

    return losses, logged_losses


def evaluate_model(model, data_loader, criterion, manager, device=device):
    model.eval()
    losses = []

    for i, batch in enumerate(data_loader):
        annIds, images, captions, caption_lengths = batch
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        for j in range(len(images)):
            image = images[j].unsqueeze(0)
            caption = captions[j]
            caption_length = caption_lengths[j]

            sampled_tokens = model.sample(image, manager)
            ann_i = manager.load_ann(annIds[j].item())
            imgId = ann_i['image_id']
            image_raw = manager.load_image(imgId)
            plt.imshow(np.array(image_raw))
            plt.show()
            # print(sampled_tokens)
            caption_j = captions[j]
            if device == torch.device('cuda'):
                sampled_tokens = sampled_tokens.cpu()
                caption_j = caption_j.cpu()
            caption_j = caption_j.numpy()
            tokens = manager.decode_tokens(sampled_tokens)
            print('predicted: ', tokens)
            print('gold:', manager.decode_tokens(caption_j, stop_at_end=True))


def compute_perplexity(model, data_loader, criterion, manager, device=device):
    """ Computes perplexity given a dataset and model. 

    Perplexity is e^(cross entropy(predicted distribution, true distribution))
    """
    token_count = 0
    model.eval()

    total_ce_loss = 0.0
    for i, batch in enumerate(data_loader):
        annIds, images, captions, caption_lengths = batch
        images = images.to(device)
        captions = captions.to(device)
        token_count += torch.sum(caption_lengths).cpu().item()

        logits = model(images, captions, caption_lengths)
        targets = pack_padded_sequence(captions, caption_lengths, batch_first=True).data
        ce_loss = criterion(logits, targets)
        total_ce_loss += ce_loss.cpu().item()

        # print(total_ce_loss, token_count)
    print('total_ce_loss', total_ce_loss)
    print('token_counts', token_count)
    pplx = np.exp(total_ce_loss/token_count)
    return pplx


if __name__ == '__main__':
    print("compute")