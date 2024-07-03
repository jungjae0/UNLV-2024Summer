import torch
import os
import pandas as pd
from glob import glob
from sklearn.utils import shuffle


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dir = './input'
path_color_dict = os.path.join(input_dir, 'color_dict.csv')
path_train_images = os.path.join(input_dir, 'train/*.jpg')
path_train_masks = os.path.join(input_dir, 'train/*.png')
color_dict = pd.read_csv(path_color_dict)
CLASSES = color_dict['name']
pd_dataset = pd.DataFrame({
    'IMAGES': sorted(glob(path_train_images)),
    'MASKS': sorted(glob(path_train_masks))
})
pd_dataset = shuffle(pd_dataset)
pd_dataset.reset_index(inplace=True, drop=True)

CFG = {
    'IMAGE_SIZE': 320,
    'BATCH_SIZE': 16,
    'EPOCHS': 10,
    'DEVICE': DEVICE,
    'CLASSES': CLASSES,
    'DATASET': pd_dataset,
    'RANDOM_STATE': 42,
    'ENCODER': 'resnet50',
    'ENCODER_WEIGHTS': 'imagenet'
}