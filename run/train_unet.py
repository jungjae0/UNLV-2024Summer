from configures import CFG
from dataloader import module_setup
from models import SegmentationModel

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPOCHS = CFG['EPOCHS']
IMAGE_SIZE = CFG['IMAGE_SIZE']


def run(dirpath, filename, model_info, random_state, batch_size, lr_key, lr_val):
    model_name = model_info['model_name']
    logger_name = model_info['logger_name']

    data_module = module_setup(random_state, batch_size)

    model = SegmentationModel(model_name, lr_val)
    
    checkpoint_callback_max = ModelCheckpoint(
        dirpath=dirpath,
        filename=f'{filename}_max',
        save_top_k=1,
        verbose=True,
        monitor="val/F1score",
        mode="max"
    )
    checkpoint_callback_min = ModelCheckpoint(
        dirpath=dirpath,
        filename=f'{filename}_min',
        save_top_k=1,
        verbose=True,
        monitor="val/F1score",
        mode="min"
    )

    logger = CSVLogger("logs-v2", name=f'{logger_name}_{random_state}_{batch_size}_{lr_key}')

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=31,
        callbacks=[checkpoint_callback_max, checkpoint_callback_min],
        max_epochs=EPOCHS,
    )

    trainer.fit(model, data_module)


def main():
    models_info = {
        'UnetPlusPlus': {'model_name': 'UnetPlusPlus', 'logger_name': 'log_unet'},
        # 'DeepLabV3Plus': {'model_name': 'DeepLabV3Plus', 'logger_name': 'landcover-classification-log-deeplab'},
        # 'SegNet': {'model_name': 'SegNet', 'logger_name': 'landcover-classification-log-segnet'}
    }

    # - UnetPlusPlus, DeepLabV3Plus, SegNet 중 돌릴 모델 이름 넣으면 됨.
    # model_info = models_info['UnetPlusPlus']
    random_states = [1, 2, 3, 4, 5]
    batch_sizes = [8]
    lr = {'lr3': 0.0001, }

#     for lr_key, lr_val in lr.items():
#         for batch_size in batch_sizes:
#             for key, model_info in models_info.items():
#                 for random_state in random_states:
#                     model_name = model_info['model_name']
#                     dirpath = 'best_checkpoints'
#                     filename = f"{model_name}_{random_state}_{batch_size}_{lr_key}"
#                     ckpt_path = os.path.join(dirpath, filename)
                    
#                     if f'{batch_size}_{lr_key}' != '16_lr3':
#                         if not os.path.exists(ckpt_path):
#                             print(f'----------Start {ckpt_path}----------')

#                             run(dirpath, filename, model_info, random_state, batch_size, lr_key, lr_val)

#                             print(f'----------Save {ckpt_path}----------')

#                         else:
#                             print(f'----------Already Save {ckpt_path}----------')
    for lr_key, lr_val in lr.items():
        for batch_size in batch_sizes:
            for key, model_info in models_info.items():
                for random_state in random_states:
                    model_name = model_info['model_name']
                    dirpath = 'best_models'
                    filename = f"{model_name}_{random_state}_{batch_size}_{lr_key}"
                    ckpt_path = os.path.join(dirpath, filename)
                    
                    # if f'{batch_size}_{lr_key}' != '16_lr3':
                    # if not os.path.exists(ckpt_path):
                    print(f'----------Start {ckpt_path}----------')

                    run(dirpath, filename, model_info, random_state, batch_size, lr_key, lr_val)

                    print(f'----------Save {ckpt_path}----------')




if __name__ == '__main__':
    main()