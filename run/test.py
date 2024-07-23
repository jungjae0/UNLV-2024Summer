from configures import CFG
from dataloader import module_setup
from models import SegmentationModel

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import os
import torch
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


EPOCHS = CFG['EPOCHS']
IMAGE_SIZE = CFG['IMAGE_SIZE']
BATCH_SIZE = CFG['BATCH_SIZE']


def run(ckpt_path, model_info, random_state, batch_size, lr_value):
    model_name = model_info['model_name']

    data_module = module_setup(random_state, batch_size)
    # ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_{random_state}.ckpt')
    # ckpt_path = os.path.join(checkpoint_dir, f'best-checkpoint-{model_name}_{random_state}.ckpt')

    # model = SegmentationModel(model_name).load_from_checkpoint(checkpoint_path=ckpt_path)
    model = SegmentationModel(model_name, lr_value)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    trainer = pl.Trainer()
    # trainer.test(model, data_module.test_dataloader())
    trainer.test(model, data_module)


    results_df = pd.DataFrame(model.result)
    results_df['random_state'] = random_state
    results_df['lr'] = lr_value

    return results_df


def main():
    checkpoint_dir = 'checkpoints'

    models_info = {
        'UnetPlusPlus': {'model_name': 'UnetPlusPlus', 'logger_name': 'landcover-classification-log-unet'},
        # 'DeepLabV3Plus': {'model_name': 'DeepLabV3Plus', 'logger_name': 'landcover-classification-log-deeplab'},
        # 'SegNet': {'model_name': 'SegNet', 'logger_name': 'landcover-classification-log-segnet'}
    }

    # lst = []
    # for random_state in random_states:
    #     df = run(checkpoint_dir, model_info, random_state)
    #     lst.append(df)
    # all_df = pd.concat(lst)
    # all_df.to_csv('result.csv', index=False)
    # print(all_df)
    
    
    
    random_states = [1]
    batch_sizes = [8]
    lr = {'lr3': 0.0001, 'lr2': 0.001}

    lst = []
    for lr_key, lr_val in lr.items():
        for batch_size in batch_sizes:
            for key, model_info in models_info.items():
                for random_state in random_states:
                    model_name = model_info['model_name']
                    dirpath = 'best_checkpoints'
                    filename = f"{model_name}_{random_state}_{batch_size}_{lr_key}.ckpt.ckpt"
                    ckpt_path = os.path.join(dirpath, filename)
                    
                    if os.path.exists(ckpt_path):
                        print(f'----------Start test {ckpt_path}----------')
                        df = run(ckpt_path, model_info, random_state, batch_size, lr_val)
                        lst.append(df)
                    else:
                        continue
    all_df = pd.concat(lst)
    all_df.to_csv("run_test.csv", index=False)

if __name__ == '__main__':
    main()