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

IMAGE_SIZE = CFG['IMAGE_SIZE']


def run(ckpt_path, model_name, random_state, batch_size, lr, result_dir):
    lr_dct = {0.001: 'lr2',  0.0001: 'lr3'}

    txt = f'test_{model_name}_{random_state}_{batch_size}_{lr_dct[lr]}.csv'


    data_module = module_setup(random_state, batch_size)
    model = SegmentationModel(model_name, lr)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    trainer = pl.Trainer()
    start_time = time.time()
    trainer.test(model, data_module)
    end_time = time.time()
    
    
    test_duration = end_time - start_time
    results_df = pd.DataFrame(model.result)
    results_df.to_csv(os.path.join(result_dir, txt), index=False)

    results_df = results_df.mean().to_frame().T
    
    results_df['model'] = model_name
    results_df['random_state'] = random_state
    results_df['batch_size'] = batch_size
    results_df['lr'] = lr
    results_df['test_time'] = test_duration

    return results_df


def main():
    result_dir = 'test_result'
    os.makedirs(result_dir, exist_ok=True)
    
    
    
    model_path_list_path = 'model_path_list.csv'
    df_models = pd.read_csv(model_path_list_path)
    df_models['lr'] = df_models['lr'].map({'lr2': 0.001, 'lr3': 0.0001})
    
    
    df_result = []
    for idx, row in df_models.iterrows():
        model_name, random_state, batch_size, lr = row['model'], row['random_state'], row['batch_size'], row['lr']
        ckpt_path = row['path']
        
        print(f'-------Start {ckpt_path}-------')
        each_result = run(ckpt_path, model_name, random_state, batch_size, lr, result_dir)
        df_result.append(each_result)
        print(f'-------Done {ckpt_path}-------')

            
    df_result = pd.concat(df_result)
    df_result.to_csv(os.path.join(result_dir, 'all_test_result.csv'), index=False)
    

if __name__ == '__main__':
    main()