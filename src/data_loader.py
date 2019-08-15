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
from src.image_processor import default_image_transform

from PIL import Image
import numpy as np

SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>"]

def collate_caption_fn(wtoi):
    def _collate_caption_fn(batch):
        batch = sorted(batch, key=lambda e: e[-1], reverse=True)
        caption_lens = torch.LongTensor([e[-1] for e in batch])
        pad_idx = wtoi['<pad>']
        max_caption_len = max(caption_lens)
        annIds       = torch.LongTensor([e[0] for e in batch])
        imgs         = torch.stack([e[1] for e in batch])
        captions     = torch.LongTensor([e[2] + [pad_idx for i in range(max_caption_len - len(e[2]))] for e in batch])
        return annIds, imgs, captions, caption_lens

    return _collate_caption_fn
    
class CaptionDataManager():
    def __init__(self, config_path, 
                 n_sample=500, 
                 shuffle=True, 
                 n_splits=[0.8, 0.1, 0.1]):

        if n_sample < 0:
            raise Exception("n_samples invalid: {} (incorrect)".format(n_sample))

        self._config = CaptionConfig(config_path)
        self._shuffle = shuffle 
        self._n_splits = n_splits
        self.n_sample = n_sample
        self.splits = {
            'val'  : [],
            'train': [],
            'test' : []
        }
        self.itow = []                    # List[index: int] -> word: str
        self.wtoi = defaultdict(int)      # Dict[word: str -> index: int]  
        self.data = defaultdict(dict)     # Dict[annId: int -> annotation: Dict]
        self.images = defaultdict(dict)   # Dict[annId: int -> image_obj: Dict]

        coco_caption =  self._config.get_coco_captions()
        
        self._build_splits(coco_caption)
        self._build_vocab(coco_caption)
        
        self._config.delete_coco()
        print("Loaded {} samples", self.n_sample)
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

    def _append_and_build_indexes(self, tokens, is_unique_tokens=False):
        if not is_unique_tokens:
            tokens = set(tokens)
            tokens = tokens.difference(self.wtoi.keys())
        wtoi_batch = {token: len(self.wtoi)+idx for idx, token in enumerate((tokens)) }
        self.wtoi = { **self.wtoi, **wtoi_batch }
        self.itow.extend(tokens)

    def _build_vocab(self, coco_caption):
        assert(coco_caption is not None)        
        self._append_and_build_indexes(SPECIAL_TOKENS, is_unique_tokens=True)

        for annId in tqdm(self.getAnnIds(), total=len(self)):
            # Load and encode annotations
            ann_i_list =  coco_caption.loadAnns(annId)
            if ann_i_list is None:
                raise Exception("ERR: skipping {}".format(annId))

            #loadAnns returns list, since we query only single object, only 1 in list.
            ann_i = ann_i_list[0] 
            self.data[annId] = ann_i
            text = ann_i['caption']
            text_processor = TextProcessor()

            processing_text = text_processor.tokenize(text, strategy="default")
            tokens = processing_text.tokens() # List[Str]

            self._append_and_build_indexes(tokens)
            self.data[annId]['tokens'] = tokens
            self.data[annId]['encoding'] = self.encode_tokens(tokens)

            # Load image objects
            imgId = self.data[annId]['image_id']
            image_i_list = coco_caption.loadImgs(imgId)
            if image_i_list is None:
                raise Exception("ERR: Cannot find imgId {} from annId {}".format(imgId, annId))

            self.images[imgId] = image_i_list[0]

    def encode_tokens(self, tokens, pad_sequence=True):
        try:
            encoding = [self.wtoi[token] for token in tokens]
            if pad_sequence:
                encoding = [self.wtoi['<start>']] + encoding + [self.wtoi['<end>']]
            return encoding
        except:
            raise Exception("cannot find some encoding for {}".format(tokens))

    
    def decode_tokens(self, encoding, length = 0, stop_at_end = False):
        tokens = []
        for i, encodingIdx in enumerate(encoding):
            encodingIdx = int(encodingIdx)
           
            if encodingIdx >= len(self.itow) or encodingIdx < 0:
                raise ValueError("no such token \'{}\' in vocabulary".format(encodingIdx))
            tokens.append(self.itow[encodingIdx])

            if length > 0 and i >= len(encoding):
                break
            if stop_at_end and encodingIdx == self.wtoi['<end>']:
                break
                
        return " ".join(tokens)

    def vocab(self):
        return self.wtoi.keys() 
    
    ''' Load annotation object {image_id: .., id:.., caption:..}
    '''
    def load_ann(self, annId):
        if annId not in self.data:
            raise ValueError("annId {} is not in dataset".format(annId))
        return self.data[annId]

    ''' Load original image object {id: <image_id>, file_path: ...}
    '''
    def load_image_object(self, imgId):
        if imgId not in self.images:
            raise ValueError("imgId {} is not in dataset".format(imgId))
        return self.images[imgId]

    ''' load image object and convert to numpy PIL image array
    '''
    def load_image(self, imgId, as_PIL = True):
        image_i =  self.load_image_object(imgId)
        image_path = self._config.get_image_path_from_object(image_i)
        img = Image.open(image_path)
        if as_PIL:
            return img
        return np.array(img)

    def getAnnIds(self, split_type=None, generate=True):
        split_keys = self.splits.keys()
        if split_type in split_keys:
            split_keys = [split_type]
        
        if not generate:
            split_arrays = [self.splits[k] for k in split_keys] #[[1,2],[3,4],..]
            annIds = [e for arr_i in split_arrays for e in arr_i] #[1,2,3,4]..
            return annIds

        def __generator_helper():
            for k in split_keys:
                for e in self.splits[k]:
                    yield e
        return __generator_helper()

    def __len__(self):
        return self.n_sample
    
    def splits_size(self):
        tr, vl, ts = self.splits['train'], self.splits['val'], self.splits['test']
        return (len(tr), len(vl), len(ts))

    def build_dataloader(self, 
                         split_type, 
                         batch_size=50, 
                         shuffle=False,
                         collate_fn=collate_caption_fn,
                         image_transform=default_image_transform):        
        dataset = CaptionDataset(self, split_type, image_transform=image_transform)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, 
                shuffle=shuffle, collate_fn=collate_caption_fn(self.wtoi))

        return dataloader
                    
''' 
CaptionDataset 
'''
class CaptionDataset(data.Dataset):
    def __init__(self, data_manager, split_type, image_transform=default_image_transform):
        if split_type not in data_manager.splits.keys():
            raise ValueError(("split_type argument must be one of" +
                    "(train, val, test), received {} instead.").format(split_type))
        
        self._image_transform = image_transform
        self._split_type = split_type
        self._data_manager = data_manager
        self.annIds = data_manager.getAnnIds(split_type=split_type, generate=False)

    def is_split_type(self, split_type):
        return self._split_type == split_type
    
    def __len__(self):
        return len(self.annIds)

    def __getitem__(self, idx):
        annId = self.annIds[idx]
        ann_i = self._data_manager.load_ann(annId)
        image = self._data_manager.load_image(ann_i['image_id'])
        image = self._image_transform(image)
        caption = ann_i['encoding']
        caption_length = len(caption)
        return annId, image, caption, caption_length

if __name__ == '__main__':
    pass


