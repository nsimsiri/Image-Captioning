import torchvision
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

default_X_norm = [0.485, 0.456, 0.406]
default_Y_norm = [0.229, 0.224, 0.225]

def validate_image(image):
    pil_arr = np.array(image)
    if len(pil_arr.shape) == 2 or pil_arr.shape[-1] == 1:
        new_pil_arr = np.full((pil_arr.shape[0], pil_arr.shape[1], 3), 255)
        expanded_pil_arr = np.expand_dims(pil_arr, (2))
        new_pil_arr[:,:,0] = pil_arr
        image = Image.fromarray(new_pil_arr.astype('uint8'), 'RGB')
    return image

def default_image_transform(image, X_norm=default_X_norm, Y_norm=default_Y_norm):
    image = validate_image(image)
    transform_f = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(X_norm, Y_norm)
    ])
    out = transform_f(image)
    return out

def train_transform(image, X_norm=default_X_norm, Y_norm=default_Y_norm):
    image = validate_image(image)
    transform_f = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(X_norm, Y_norm)
    ])
    out = transform_f(image)
    return out

