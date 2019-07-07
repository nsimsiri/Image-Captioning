import sys, os, json, yaml

import torch
from torch.utils import data
from torchvision import transforms
from sklearn.model_selection import train_test_split

from pycocotools.coco import Coco

from src.config import CaptionConfig

from PIL import Image
import numpy as np

class CaptionDataset():
    def __init__(self, config_path, 
                 n_sample=500, 
                 shuffle=True, 
                 n_splits=[0.8, 0.1, 0.1]):

        if n_sample < 0:
            raise Exception("n_samples invalid: {} (incorrect)".format(n_sample))

        self._config = CaptionConfig(config_path)
        self._shuffle = shuffle 
        self._n_splits = n_splits
        self.n_sample
        self.splits = {
            'val': None,
            'train': None,
            'test': None
        }

        coco_caption =  self._config.get_coco_captions()

    def _get_coco_annotation_ids(self, coco_caption):
        annIds = coco_caption.getAnnIds()
        if not self._shuffle: 
            return annIds
        annIds = np.random.choice(annIds, size=self.n_sample, replace=False)
        return annIds

    def _get_splits(self, coco_caption):
        if len(self._n_splits) != 3 or sum(self._n_splits)!=1:
            raise Exception("Splits %'s are not valid: {} (incorrect)".format(splits))

        annIds = self._get_coco_annotation_ids()
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
                            test_size =float(test_size/val_train_percent))


        self.splits['train'] = train_annIds
        self.splits['val']   = val_annIds
        self.splits['test']  = test_annIds

    def __len__(self):
        return self.n_sample
    
    def splits_size(self):
        tr, vl, ts = self.splits['train'], self.splits['val'], self.splits['test']
        return (len(tr), len(vl), len(ts))

if __name__ == '__main__':
    pass


