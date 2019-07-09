import sys, os, json, yaml
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict

import torch
from torch.utils import data
from torchvision import transforms
from sklearn.model_selection import train_test_split

from pycocotools.coco import COCO
import nltk
from nltk import word_tokenize

from src.config import CaptionConfig
from src.text_processor import TextProcessor
from src.image_processor import default_image_transformer

from PIL import Image
import numpy as np

SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>"]

def collate_caption_fn(batch):
    pass

class CaptionDataManager():
    def __init__(self, config_path, 
                 n_sample=500, 
                 shuffle=True, 
                 n_splits=[0.8, 0.1, 0.1],
                 image_transform = default_image_transform):

        if n_sample < 0:
            raise Exception("n_samples invalid: {} (incorrect)".format(n_sample))

        self._config = CaptionConfig(config_path)
        self._shuffle = shuffle 
        self._n_splits = n_splits
        self.n_sample = n_sample
        self.image_transform = image_transform
        self.splits = {
            'val'  : [],
            'train': [],
            'test' : []
        }
        self.itow = []                    # List[index: int] -> word: str
        self.wtoi = defaultdict(int)      # Dict[word: str -> index: int]  
        self.data = defaultdict(dict)     # Dict[annId: int -> annotation: Dict]

        coco_caption =  self._config.get_coco_captions()
        
        self._build_splits(coco_caption)
        self._build_vocab(coco_caption)
        del coco_caption

    def _get_coco_annotation_ids(self, coco_caption):
        annIds = coco_caption.getAnnIds()
        if not self._shuffle: 
            return annIds
        annIds = np.random.choice(annIds, size=self.n_sample, replace=False).tolist()
        return annIds

    def _build_splits(self, coco_caption):
        if len(self._n_splits) != 3 or sum(self._n_splits)!=1:
            raise Exception("Splits %'s are not valid: {} (incorrect)".format(splits))

        annIds = self._get_coco_annotation_ids(coco_caption)
        n_annIds = len(annIds)
        train_percent, val_percent, test_percent = self._n_splits
        val_train_percent = val_percent + test_percent
        train_annIds, test_annIds =\
            train_test_split(annIds,\
                            train_size=train_percent,\
                            test_size=val_percent + test_percent)

        val_annIds, test_annIds =\
            train_test_split(test_annIds,\
                            train_size=float(val_percent/val_train_percent),\
                            test_size =float(test_percent/val_train_percent))


        self.splits['train'] = train_annIds
        self.splits['val']   = val_annIds
        self.splits['test']  = test_annIds

    def _append_and_build_indexes(self, tokens):
        tokens = set(tokens)
        tokens = tokens.difference(self.wtoi.keys())
        wtoi_batch = {token: len(self.wtoi)+idx for idx, token in enumerate((tokens)) }
        self.wtoi = { **self.wtoi, **wtoi_batch }
        self.itow.extend(tokens)

    def _build_vocab(self, coco_caption):
        assert(coco_caption is not None)        
        self._append_and_build_indexes(SPECIAL_TOKENS)

        for annId in tqdm(self.getAnnIds(), total=len(self)):
            ann_i_list =  coco_caption.loadAnns(annId)
            if ann_i_list is None:
                raise Exception("ERR: skipping {}".format(annId))

            ann_i = ann_i_list[0]
            self.data[annId] = ann_i
            text = ann_i['caption']
            text_processor = TextProcessor()

            processing_text = text_processor.tokenize(text, strategy="default")
            tokens = processing_text.tokens() # List[Str]

            self._append_and_build_indexes(tokens)
            self.data[annId]['tokens'] = tokens
            self.data[annId]['encoding'] = self.encode_tokens(tokens)

    def encode_tokens(self, tokens, pad_sequence=True):
        try:
            encoding = [self.wtoi[token] for token in tokens]
            if pad_sequence:
                encoding = [self.wtoi['<start>']] + encoding + [self.wtoi['<end>']]
            return encoding
        except:
            raise Exception("cannot find some encoding for {}".format(tokens))

    def load_image(self, annId):
        ann_i = self.load_ann(annId)
        

    def vocab(self):
        return self.wtoi.keys() 
    
    def load_ann(self, annId):
        if annId not in self.data:
            raise ValueError("annId {} is not in dataset".format(annId))
        return self.data[annId]

    def getAnnIds(self, generate=True):
        split_keys = ['train', 'val', 'test']
        if not generate:
            return [e for e in [self.splits[k] for k in split_keys]]

        for k in split_keys:
            for e in self.splits[k]:
                yield e

    def __len__(self):
        return self.n_sample
    
    def splits_size(self):
        tr, vl, ts = self.splits['train'], self.splits['val'], self.splits['test']
        return (len(tr), len(vl), len(ts))


class CaptionDataset(data.Dataset):
    def __init__(self, data_manager, split_type, ):
        if split_type not in data_manager.splits.keys():
            raise ValueError(("split_type argument must be one of" +
                    "(train, val, test), received {} instead.").format(split_type))
        
        self._split_type = split_type
        self._data_manager = data_manager

    def is_split_type(self, split_type):
        return self._split_type == split_type
    
    def __len__(self):
        
        return len(annIds)

    def __getitem__(self, idx):
        annIds = self._data_manager[self._split_type]
        annId = annIds[idx]
        ann_i = self._data_manager.load_ann(annId)
        image_path = self._data_manager.load_image(annId)

        image = Image.open(image_path)
        caption = ann_i['encoding']

        return image, caption




if __name__ == '__main__':
    pass


