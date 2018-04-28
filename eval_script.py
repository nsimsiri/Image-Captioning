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
import time

class Args(object):
    def __init__(self, img_file_name, folder='val2014',hardpath=None):
        img_location = './coco/%s/%s'%(folder,img_file_name);
        if hardpath!=None:
            img_location = hardpath+"/"+img_file_name;
#         self.encoder_path = './models/encoder-5-3000.pkl'
#         self.decoder_path = './models/decoder-5-3000.pkl'
        self.encoder_path = './models/1_256_512_encoder-5-1001.pkl'
        self.decoder_path = './models/1_256_512_decoder-5-1001.pkl'
        self.vocab_path = './data/vocab.pkl'
        self.embed_size = 256
        self.hidden_size = 512;
        self.num_layers = 1;
        self.image = img_location;
        self.encoder = None;
        self.decoder = None;
        self.vocab = None;


dataDir='./coco'
VAL = 'val2014';
TR = 'train2014';
# dataType='val2014'
dataType='train2014'
annFile='{}/annotations/sm_captions_{}.json'.format(dataDir,dataType)
print annFile
coco = COCO(annFile)
imgIds = coco.getImgIds()
print 'Annotations', len(coco.getAnnIds());
print 'Imags', len(imgIds)
img = coco.loadImgs(imgIds[2])[0]
args = Args(img['file_name'], hardpath=('./data/val_resized2014' if dataType==VAL else './data/resized2014'));
coco.dataset['type'] = None;
annIds = coco.getAnnIds(imgIds=[img['id']])
ann = coco.loadAnns(annIds)
print 'img-id', ann[0]['image_id']
test = [x['caption'] for x in ann]
print '\n------TRUE CAPTIONS------';
for sen in test:
    print sen

# caption generator
a_time = time.time()
print '--results--\n'
caption, encoder, decoder, vocab = main(args, show_img=False)
print caption
caption = caption.replace('<start>','').replace('end>','')
print time.time()-  a_time
print caption;
