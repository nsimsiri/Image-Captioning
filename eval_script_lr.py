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
GENCAP_DIR = './gen_caps/GRU_EXP_LR.pkl'
XODER_PATH = './models/GRU_LR/'
dataDir='./coco'
mypath = "./data/val_resized2014"
# mypath = "./data/resized2014"
VAL = 'val2014';
TR = 'train2014';
GEN_CAP_DIR = './gen_caps'
dataType='val2014'
# dataType='train2014'
annFile='{}/annotations/sm_captions_{}.json'.format(dataDir,dataType)
print 'running eval_script_lr on caption=%s xoder=%s dataType=%s annFile=%s'%(GENCAP_DIR, XODER_PATH, dataType, annFile)
# coco = COCO(annFile)
# imgIds = coco.getImgIds()
# print 'Annotations', len(coco.getAnnIds());
# print 'Imags', len(imgIds)

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
        self.embed_size = 128 if IS_GRU() else 256;
        self.hidden_size = 256 if IS_GRU() else 512;
        self.num_layers = 1;
        self.image = img_location;
        self.encoder = None;
        self.decoder = None;
        self.vocab = None;
        self.use_gru = _IS_GRU;
    def __str__(self):
        return 'Arg[%s img_loc=%s, enc=%s, dec=%s, emb=%s, hid=%s, nlayer=%s]'%(\
                'GRU' if IS_GRU() else 'LSTM', self.image, self.encoder_path, \
                self.decoder_path, self.embed_size, self.hidden_size, self.num_layers);

# arg2xoder = defaultdict(dict);
# stub = ['learning_00001_decoder-5-1001.pkl', 'learning_0001_encoder-5-1001.pkl','learning_01_decoder-5-1001.pkl','learning_10_encoder-5-1001.pkl',
# 'learning_00001_encoder-5-1001.pkl',  'learning_001_decoder-5-1001.pkl',   'learning_01_encoder-5-1001.pkl',  'learning_1e-05_decoder-5-1001.pkl',
# 'learning_0001_decoder-5-1001.pkl',   'learning_001_encoder-5-1001.pkl',   'learning_10_decoder-5-1001.pkl',  'learning_1e-05_encoder-5-1001.pkl']
# for fn in stub:
#     for xoder in ['decoder', 'encoder']:
#         if (xoder in fn):
#             arg = '_'.join(fn.split('_')[:2]);
#             # if 'e' not in arg:
#             #     l = list(arg)
#             #     l.insert(1,'.')
#             #     arg = ''.join(l);
#             print arg;
#             arg2xoder[tuple(arg)][xoder] = fn;
#
# sys.exit()

for (dirpath, dirnames, filenames) in walk(XODER_PATH):
    arg2xoder = defaultdict(dict);
    for fn in filenames:
        for xoder in ['decoder', 'encoder']:
            if (xoder in fn):
                arg = '_'.join(fn.split('_')[:2])
                arg2xoder[arg][xoder] = fn;

    # for each models
    EVAL_MAP = {};
    cached_vocab = None;
    for arg, xoder2fn in arg2xoder.iteritems():
        print '-------';
        GEN_CAPS = []
        cached_encoder = None;
        cached_decoder = None;
        print arg
        print 'Evaluating model %s'%(arg);
        for img_id in imgIds:

            print '--\nevaluating img: ', img_id
            try :
                img = coco.loadImgs(img_id)[0] # index 0 because only 1 img return ie. coco.loadImgs.. = [img_we_want]
                args = Args(img['file_name'], hardpath=('./data/val_resized2014' if dataType==VAL else './data/resized2014'), \
                            enc=xoder2fn['encoder'], dec=xoder2fn['decoder']);
                coco.dataset['type'] = None; # to fix bug with diff versio.
                annIds = coco.getAnnIds(imgIds=[img['id']])
                ann = coco.loadAnns(annIds)
                gold = [x['caption'] for x in ann]
                caption_obj = copy.copy(ann[0]);
                caption_obj['file_name'] = str(args.image); #file name'

                args.encoder = cached_encoder;
                args.decoder = cached_decoder;
                args.vocab = cached_vocab;
                caption, cached_encoder, cached_decoder, cached_vocab = main(args, show_img=False);
                print 'caption', caption;
                caption = caption.replace('<start>','').replace('<end>','')
                caption_obj['caption'] = caption;
                GEN_CAPS.append(caption_obj);
            except Exception as e:
                print '\n-- error on img_id', img_id,'\n',e,'\n';

        EVAL_MAP[arg] = GEN_CAPS;
        break;

    print 'Dumping EVAL_MAP json...';
    with open(GENCAP_DIR, 'wb') as handle:
        print 'writing to ', GENCAP_DIR;
        pickle.dump(EVAL_MAP, handle)
    print 'done';
# caption generator
# a_time = time.time()
# print '--results--\n'
# caption, encoder, decoder, vocab = main(args, show_img=False)
# print caption
# caption = caption.replace('<start>','').replace('end>','')
# print time.time()-  a_time
# print caption;
