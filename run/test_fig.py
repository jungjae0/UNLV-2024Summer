from configures import CFG
from dataloader import module_setup
from models import SegmentationModel

# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import os
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = CFG['IMAGE_SIZE']
color_dict = pd.read_csv('input/class_dict.csv')
CLASSES = color_dict['name']

def rgb2category(rgb_mask):
    category_mask = np.zeros(rgb_mask.shape[:2], dtype=np.int8)
    for i, row in color_dict.iterrows():
        category_mask += (np.all(rgb_mask.reshape((-1, 3)) == (row['r'], row['g'], row['b']), axis=1).reshape(rgb_mask.shape[:2]) * i)
    return category_mask

def category2rgb(category_mask):
    rgb_mask = np.zeros(category_mask.shape[:2] + (3,))
    for i, row in color_dict.iterrows():
        rgb_mask[category_mask==i] = (row['r'], row['g'], row['b'])
    return np.uint8(rgb_mask)


def run(model_name, ckpt_path):
    lr = '0.0001'
    batch_size = 16
    random_state = 3
    model = SegmentationModel(model_name, lr)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    data_module = module_setup(random_state, batch_size)
    image, mask = next(iter(data_module.test_dataloader()))
    outputs = model(image)
    for i in range(batch_size//2):
        pred_category = outputs[i].argmax(dim=0).unsqueeze(1).type(torch.int64).squeeze()
        pred_rgb = category2rgb(pred_category.cpu())
        gold_mask = category2rgb(mask[i].squeeze().cpu())
        test_img = image[i].permute(1,2,0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10))
        ax1.set_title('IMAGE')
        ax1.imshow(test_img)

        ax2.set_title('MASK')
        ax2.imshow(gold_mask)

        ax3.set_title('PRED')
        ax3.imshow(pred_rgb)
        
        # plt.suptitle(f'Best {model_name} - Comparison of Image, Mask, and Prediction')
        
        png_path = f'codes/best_png/{model_name}_best_{i}.png'
        plt.savefig(png_path)
        
        
        
        
def main():
 
    best_models = {'DeepLabV3Plus':'best_checkpoints/DeepLabV3Plus_3_16_lr3_max.ckpt',
                  'UnetPlusPlus': 'best_models/UnetPlusPlus_3_16_lr3_max.ckpt'}
    
    for model_name, ckpt_path in best_models.items():
        run(model_name, ckpt_path)
    

if __name__ == '__main__':
    main()