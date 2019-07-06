import sys, os, json
from collections import defaultdict
import yaml
from yaml import CLoader as Loader
from pycocotools.coco import COCO

def init_configs(config_path):
    try:
        assert(os.path.exists(config_path))
    except Exception as e:
        print(e)
    config_stream = open(config_path, 'r')
    config = yaml.load(config_stream, Loader=Loader)
    return config;

class CaptionConfig():
    
    def __init__(self, config_path, filters=['instances']):
        self.config_path = config_path
        config_yml = init_configs(config_path)
        coco_yml = config_yml['config']['data']['coco']
        self.filters = filters
        self.data_path = defaultdict(dict)
        self.coco = defaultdict(dict)

        for split in ['val', 'train', 'test']:
            for data_type in ['images', 'captions', 'instances']:
                _path = None
                if data_type is 'images':
                    _path = os.path.join(coco_yml['root'], coco_yml[split][data_type])
                else:    
                    _path = os.path.join(coco_yml['root'], coco_yml['caption_ann'], coco_yml[split][data_type])    
                try:
                    assert(os.path.exists(_path))
                    self.data_path[split][data_type] = _path
                except Exception as e:
                    print(e)

            if (split in self.data_path):
                for data_type in ['captions' ,'instances']:
                    if self._skip(data_type): 
                        continue
                    self.coco[split][data_type] = COCO(self.data_path[split][data_type])
                    print("loaded - {} {}".format(split, data_type))
    
    def _skip(self, e):
        return e in self.filters

    def get_images_folder_path(self, split='val'):
        if (split not in ['val', 'test', 'train']):
            raise Exception("Split is invalid");

        return self.data_path[split]['images']

    def get_image_path(self, imgIds, split='val', data_type='captions'):
        if split not in ['val', 'test', 'train']:
            raise Exception("Split is invalid")
        
        coco_obj = self.coco[split][data_type] 
        imgObjs =  coco_obj.loadImgs(imgIds)
        image_folder_path = self.get_images_folder_path(split=split)

        def img_obj_to_path(img_obj):
            imgName = img_obj['file_name']
            _path = os.path.join(image_folder_path, imgName)
            return _path

        image_paths = [img_obj_to_path(imgObj) for imgObj in imgObjs]
        return image_paths


    def get_coco_captions(self, split='val'):
        if (split not in ['val', 'test', 'train']):
            raise Exception("Split is invalid");
        
        return self.coco[split]['captions']
    

    def get_coco_instances(self, split='val'):
        if self._skip('instances'):
            raise Exception("instances is not loaded.")
        if (split not in ['val', 'test', 'train']):
            raise Exception("Split is invalid");
        
        return self.coco[split]['instances']
