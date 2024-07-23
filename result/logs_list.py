import os
import pandas as pd
import tqdm

# logs deeplab-lr3와 unet-lr2가 들어있음
# logs-v2 deeplab-lr2와 unet-lr3가 들어있고
# 예외적으로 best_models에 deeplab-lr3-1-8이 들어있음


logs_dir = 'logs'
logs_v2_dir = 'logs-v2'


def org_log_df(root, file, df_all, paths, model):
    metric_path = os.path.join(root, file)
    df = pd.read_csv(metric_path)
    if 48 in df['epoch'].values:
        point = root.split('/')[1].split('log_')[1]
        model = point.split('_')[0]
        random_state = point.split('_')[1]
        batch_size = point.split('_')[2]
        lr = point.split('_')[3]
        grouped_columns = df.columns[2:]
        df_grouped = df.groupby([df.columns[0], df.columns[1]])[grouped_columns].sum().reset_index()
        txt = f'{random_state}_{batch_size}_{lr}'
        df_grouped['comb'] = model + '_' + txt
        df_all.append(df_grouped)
        paths.append(metric_path)
        return df_grouped

df_all = []

paths = []

# 1. logs 정리
for root, dirs, files in (os.walk(logs_dir)):
    for file in files:
        if file.endswith(".csv") and not 'checkpoint' in file:
            if 'deep' in root and 'lr3' in root: 
                org_log_df(root, file, df_all, paths, 'deep')
            elif 'unet' in root and 'lr2' in root:
                org_log_df(root, file, df_all, paths, 'unet')


for root, dirs, files in (os.walk(logs_v2_dir)):
    for file in files:
        if file.endswith(".csv") and not 'checkpoint' in file:
            if 'deep' in root and 'lr2' in root: 
                org_log_df(root, file, df_all, paths, 'deep')
            elif 'unet' in root and 'lr3' in root:
                org_log_df(root, file, df_all, paths, 'unet')
            elif 'deep' in root and '1_8_lr3' in root:
                org_log_df(root, file, df_all, paths, 'deep')
                    
df_all = pd.concat(df_all)  
df_all.to_csv('all_logs.csv', index=False)
