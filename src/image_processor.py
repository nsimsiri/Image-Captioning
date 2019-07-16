import torchvision
from torchvision import transforms, utils
import numpy as np

import warnings
warnings.filterwarnings('ignore')

default_X_norm = [0.485, 0.456, 0.406]
default_Y_norm = [0.229, 0.224, 0.225]

def default_image_transform(image, X_norm=default_X_norm, Y_norm=default_Y_norm):
    transform_f = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(X_norm, Y_norm)
    ])
    return transform_f(image)

def train_transform(image, X_norm=default_X_norm, Y_norm=default_Y_norm):
    transform_f = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(X_norm, Y_norm)
    ])
    return transform_f(image)

