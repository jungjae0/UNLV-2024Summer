import os
import itertools

import pandas as pd


# best_models에는 deeplab-lr2와 unet-lr3가 들어있고
# best_checkpoints에는 deeplab-lr3와 unet-lr2가 들어있음
# 예외적으로 best_models에 deeplab-lr3-1-8이 들어있음

random_states = list(range(1, 6))
lrs = ['lr2', 'lr3']
batch_sizes = [8, 16]

combinations = list(itertools.product(random_states, lrs, batch_sizes))

df_comb = pd.DataFrame(combinations, columns=['random_state', 'lr', 'batch_size'])

print(len(df_comb))

best_models_dir = 'best_models'
best_ckpts_dir = 'best_checkpoints'
deep_lr2, deep_lr3 = [], []
unet_lr2, unet_lr3 = [], []


# 1. best_models 정리
best_models_files = [path for path in os.listdir(best_models_dir) if 'max' in path]
for file in best_models_files:
    path = os.path.join(best_models_dir, file)
    if 'Unet' in file and 'lr3' in file:
        unet_lr3.append(path)
    elif 'Deep' in file and 'lr2' in file:
        deep_lr2.append(path)
    # elif 'Deep' in file and 'lr3' in file and '8' in file:
    #     deep_lr3.append(path)
        
# 2. best_checkpoints 정리
best_ckpts_files = [path for path in os.listdir(best_ckpts_dir) if 'max' in path]
    
for file in best_ckpts_files:
    path = os.path.join(best_ckpts_dir, file)
    if 'Unet' in file and 'lr2' in file:
        unet_lr2.append(path)
    elif 'Deep' in file and 'lr3' in file:
        if 'DeepLabV3Plus_1_8_lr3' in file:
            if '-v' in file:
                deep_lr3.append(path)
            else:
                continue
        else:
            deep_lr3.append(path)
       
    
    
all_path = deep_lr2 + deep_lr3 + unet_lr2 + unet_lr3
df_all = pd.DataFrame({'path': all_path})
df_all['file'] = df_all['path'].str.split('/').str[1]
df_all['model'] = df_all['file'].str.split('_').str[0]
df_all['random_state'] = df_all['file'].str.split('_').str[1]
df_all['batch_size'] = df_all['file'].str.split('_').str[2]
df_all['lr'] = df_all['file'].str.split('_').str[3]


df_all.to_csv('model_path_list.csv', index=False)


# data = {'deep_lr2': deep_lr2, 'deep_lr3': deep_lr3, 'unet_lr2': unet_lr2, 'unet_lr3': unet_lr3}
# df_model = pd.DataFrame(data)
# print(df_model)
                
        
        

# for idx, row in df_comb.iterrows():
#     txt = f"{row['random_state']}_{row['batch_size']}_{row['lr']}"






# df_comb['txt'] = df['random_state'].astype(str) + '_' + df['lr'] + '_' + df['batch_size'].astype(str)







# for idx, row in df_comb.iterrows():
#     txt = f"{row['random_state']}_{row['batch_size']}_{row['lr']}"
#     for model_dir in model_dirs:
#         for file in os.listdir(model_dir):
            
        
    
    # txt = row['random_state'].astype(str) + '_' + row['lr'] + '_' + row['batch_size'].astype(str)
    # print(txt)
                                                          
                                                         

# unet_max, unet_min, deep_max, deep_min = [], [], [], []



# model_dirs = ['best_models']

# for model_dir in model_dirs:
#     for file in os.listdir(model_dir):
#         path = os.path.join(model_dir, file)
#         if 'Unet' in path:
#             if not 'v' in path:
#                 if 'max' in path:
#                     unet_max.append(path)
#                 else:
#                     unet_min.append(path)
#         elif 'Deep' in path:
#             if not '8_lr3' in path:

#                 if 'max' in path:
#                     deep_max.append(path)
#                 else:
#                     deep_min.append(path)
                
# dct = {'unet_max': unet_max, 'unet_min': unet_min, 'deep_max': deep_max, 'deep_min': deep_min}        
# # dct = {'unet_max': unet_max, 'unet_min': unet_min}               

# # for key, value in dct.items():
# #     print(key, len(value))
    

# df = pd.DataFrame(dct)
# print(df)


