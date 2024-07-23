import cv2
import numpy as np
import albumentations as aug

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from configures import CFG
import os
import pandas as pd

color_dict = CFG['COLOR_DICT']
pd_dataset = CFG['DATASET']
IMAGE_SIZE = CFG['IMAGE_SIZE']
num_workers = 1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#
# pd_dataset = shuffle(pd_dataset)
# pd_dataset.reset_index(inplace=True, drop=True)
#
# pd_train, pd_test = train_test_split(pd_dataset, test_size=0.10, random_state=CFG['RANDOM_STATE'])
# pd_train, pd_val = train_test_split(pd_train, test_size=0.10, random_state=CFG['RANDOM_STATE'])

#data_root = 'random_data'
#random_state = CFG['RANDOM_STATE']

#pd_train = pd.read_csv(os.path.join(data_root, f'train_{random_state}.csv'))
#pd_val = pd.read_csv(os.path.join(data_root, f'val_{random_state}.csv'))
#pd_test = pd.read_csv(os.path.join(data_root, f'test_{random_state}.csv'))

#train_augment = aug.Compose([
#    aug.Resize(IMAGE_SIZE, IMAGE_SIZE),
#    aug.HorizontalFlip(p=0.5),
#    aug.VerticalFlip(p=0.5),
#    aug.RandomBrightnessContrast(p=0.3)
#])

#test_augment = aug.Compose([
#    aug.Resize(IMAGE_SIZE, IMAGE_SIZE),
#    aug.RandomBrightnessContrast(p=0.3)
#])


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


class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image = cv2.imread(row.IMAGES)
        image = image[:, :, ::-1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row.MASKS)
        # mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, ::-1]

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        mask = rgb2category(mask)

        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        mask = np.expand_dims(mask, axis=0)

        image = torch.Tensor(image) / 255.0
        mask = torch.Tensor(mask).long()

        return image, mask



class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, pd_train, pd_val, pd_test,train_augment, test_augment, batch_size=10):
        super().__init__()
        self.pd_train = pd_train
        self.pd_val = pd_val
        self.pd_test = pd_test
        self.train_augment=train_augment
        self.test_augment=test_augment
        self.batch_size=batch_size

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(self.pd_train, self.train_augment)
        self.val_dataset = SegmentationDataset(self.pd_val, self.test_augment)
        self.test_dataset = SegmentationDataset(self.pd_test, self.test_augment)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size // 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size // 2, shuffle=False, num_workers=num_workers, pin_memory=True)


#def module_setup():
#    data_module = SegmentationDataModule(pd_train, pd_val, pd_test, batch_size=CFG['BATCH_SIZE'])
#    data_module.setup()

#    return data_module

def module_setup(random_state, batch_size, n_splits=5):
    data_root = 'random_data'

    pd_train = pd.read_csv(os.path.join(data_root, f'train_split_{random_state}.csv'))
    pd_val = pd.read_csv(os.path.join(data_root, f'val_split_{random_state}.csv'))
    pd_test = pd.read_csv(os.path.join(data_root, f'test_split_{random_state}.csv'))

    train_augment = aug.Compose([
        aug.Resize(IMAGE_SIZE, IMAGE_SIZE),
        aug.HorizontalFlip(p=0.5),
        aug.VerticalFlip(p=0.5),
        aug.RandomBrightnessContrast(p=0.3)
    ])

    test_augment = aug.Compose([
        aug.Resize(IMAGE_SIZE, IMAGE_SIZE),
        aug.RandomBrightnessContrast(p=0.3)
    ])


    data_module = SegmentationDataModule(pd_train, pd_val, pd_test, train_augment, test_augment, batch_size=batch_size)
    data_module.setup()
    return data_module

