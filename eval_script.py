from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
from sample import *
import torch
import argparse
from PIL import Image
import pickle
import os
from os import walk
import time
import sys;
from collections import defaultdict
import json
import copy;
import pickle

_IS_GRU = 1
IS_GRU = (lambda: _IS_GRU == 1);
GENCAP_DIR = './gen_caps/GRU_EXP.pkl'
XODER_PATH = './models/'
dataDir='./coco'
mypath = "./data/val_resized2014"
# mypath = "./data/resized2014"
VAL = 'val2014';
TR = 'train2014';
GEN_CAP_DIR = './gen_caps'
dataType='val2014'
# dataType='train2014'
annFile='{}/annotations/sm_captions_{}.json'.format(dataDir,dataType)
print annFile
coco = COCO(annFile)
imgIds = coco.getImgIds()
print 'Annotations', len(coco.getAnnIds());
print 'Imags', len(imgIds)

class Args(object):
    def __init__(self, img_file_name, folder='val2014',hardpath=None, enc=None, dec=None):
        img_location = './coco/%s/%s'%(folder,img_file_name);
        if hardpath!=None:
            img_location = hardpath+"/"+img_file_name;
#         self.encoder_path = './models/encoder-5-3000.pkl'
#         self.decoder_path = './models/decoder-5-3000.pkl'
        self.encoder_path = XODER_PATH + enc#'./models/1_256_512_encoder-5-1001.pkl'
        self.decoder_path = XODER_PATH + dec#'./models/1_256_512_decoder-5-1001.pkl'
        self.vocab_path = './data/vocab.pkl'
        self.embed_size = 256
        self.hidden_size = 512;
        self.num_layers = 1;
        self.image = img_location;
        self.encoder = None;
        self.decoder = None;
        self.vocab = None;
        self.use_gru = IS_GRU;
    def __str__(self):
        return 'Arg[%s img_loc=%s, enc=%s, dec=%s, emb=%s, hid=%s, nlayer=%s]'%(\
                'GRU' if IS_GRU() else 'LSTM', self.image, self.encoder_path, \
                self.decoder_path, self.embed_size, self.hidden_size, self.num_layers);

for (dirpath, dirnames, filenames) in walk(mypath):
    pass;

for (dirpath, dirnames, filenames) in walk(XODER_PATH):
    arg2xoder = defaultdict(dict);
    for fn in filenames:
        for xoder in ['decoder', 'encoder']:
            if (xoder in fn):
                arg = filter(lambda x: len(x.strip())!=0, fn.split(xoder)[0].split("_"))
                arg2xoder[tuple(arg)][xoder] = fn;

    # for each models
    EVAL_MAP = {};
    cached_vocab = None;
    for arg, xoder2fn in arg2xoder.iteritems():
        print '-------';
        num_layers = arg[0]
        embed_size = arg[1];
        hidden_size = arg[2];
        GEN_CAPS = []
        cached_encoder = None;
        cached_decoder = None;

        print 'Evaluating model - nlayer:% s emb:%s hid:%s'%(num_layers, embed_size, hidden_size);
        for img_id in imgIds:

            print '--\nevaluating img: ', img_id
            try :
                img = coco.loadImgs(img_id)[0] # index 0 because only 1 img return ie. coco.loadImgs.. = [img_we_want]
                args = Args(img['file_name'], hardpath=('./data/val_resized2014' if dataType==VAL else './data/resized2014'), \
                            enc=xoder2fn['encoder'], dec=xoder2fn['decoder']);
                args.num_layers = int(num_layers);
                args.embed_size = int(embed_size);
                args.hidden_size = int(hidden_size);
                coco.dataset['type'] = None; # to fix bug with diff versio.
                annIds = coco.getAnnIds(imgIds=[img['id']])
                ann = coco.loadAnns(annIds)
                gold = [x['caption'] for x in ann]
                caption_obj = copy.copy(ann[0]);
                caption_obj['file_name'] = str(args.image); #file name'
                # print '\n------TRUE CAPTIONS------';
                # for sen in gold:
                #     print sen
                args.encoder = cached_encoder;
                args.decoder = cached_decoder;
                args.vocab = cached_vocab;
                caption, cached_encoder, cached_decoder, cached_vocab = main(args, show_img=False);
                print 'caption', caption;
                caption = caption.replace('<start>','').replace('<end>','')
                caption_obj['caption'] = caption;
                # img['caption'] = caption;
                # gen_captions.append({'caption': caption, 'image_id': img_id});
                GEN_CAPS.append(caption_obj);
            except Exception as e:
                print '\n-- erorr on img_id', img_id,'\n',e,'\n';

        EVAL_MAP[arg] = GEN_CAPS;

    print 'Dumping EVAL_MAP json...';
    with open(GENCAP_DIR, 'wb') as handle:
        print 'writing to ', GENCAP_DIR;
        pickle.dump(EVAL_MAP, handle)
    print 'done';
