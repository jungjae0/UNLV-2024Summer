import plotly.express as px
import pandas as pd
import plotly.io as pio
import os
from glob import glob
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import numpy as np



# df = pd.read_csv('../random_data/train_split_1.csv')
# cols = [col for col in df.columns if col != 'IMAGES' or col != 'MASKS']


# fig = px.parallel_categories(
#     df[cols], 
#     # color="forest_land", 
#     color_continuous_scale="sunset",
#     title="Parallel categories plot of targets"
#     )

# pio.write_image(fig, "image.png")

dataPath = "input"
os.chdir(dataPath)
IMAGE_SIZE = 320
BATCH_SIZE = 7
EPOCHS = 25

color_dict = pd.read_csv('class_dict.csv')
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

def getCats(path):
    mask = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    mask = np.array(mask)
    mask = rgb2category(mask)
    return mask

def processMask(df):
    for cls in CLASSES:
        df[cls] = 0

    for i in tqdm(range(len(df))):
        mask = getCats(df['MASKS'][i])
        onehot = [0]*7
        for idx in np.unique(mask):
            onehot[idx]=1
        df.iloc[i, 2:] = onehot
    return df  

pd_dataset = pd.DataFrame({
    'IMAGES': sorted(glob("train/*.jpg")), 
    'MASKS': sorted(glob("train/*.png"))
})
pd_dataset = shuffle(pd_dataset)
pd_dataset.reset_index(inplace=True, drop=True)

df = processMask(pd_dataset)

clsdf = [
    ['urban_land', df['urban_land'].sum()],
    ['agriculture_land', df['agriculture_land'].sum()],
    ['rangeland', df['rangeland'].sum()],
    ['forest_land', df['forest_land'].sum()],
    ['water', df['water'].sum()],
    ['barren_land', df['barren_land'].sum()],
    ['unknown', df['unknown'].sum()],
]

clsdf = pd.DataFrame(clsdf, columns=["Category", "Instances"])

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(x = 'Category',
            y = 'Instances',
            data = clsdf,)
            # errwidth=0)
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 10.5, clsdf['Instances'][i],ha="center")
    
plt.savefig('data_split.png')