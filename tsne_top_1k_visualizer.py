import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
import random
from os import path
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

#EMB_DIR = f'D:\\glove_fasttext\\glove.840B.300d'
EMB_DIR = f'D:\\glove_fasttext\\crawl-300d-2M.vec'

#EMB_FILE =f'glove.840B.300d.txt'
EMB_FILE = f'crawl-300d-2M.vec'

#TOP_100K_FILE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_100k.csv'
TOP_100K_FILE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_100k.csv'

embedding_100k = pd.read_csv(TOP_100K_FILE, na_values=None, keep_default_na=False,names=['word','female_effect_size','p_value'],skiprows=1)
top_1k = embedding_100k.head(1000)
target_words = top_1k['word'].tolist()

#Skip row for FT
#embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=1000)
embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=1000,skiprows=1)

target_df = embedding_df.loc[target_words]
target_data = target_df.to_numpy()
print(target_data.shape)

reduced_dims = TSNE().fit_transform(target_data)
tsne_df = pd.DataFrame(reduced_dims,index=target_words,columns=['x','y'])
tsne_df.to_csv(path.join(EMB_DIR,f'tsne_dims_1k.csv'))
print(tsne_df)

tsne_x = tsne_df['x'].tolist()
tsne_y = tsne_df['y'].tolist()
top_es = top_1k['female_effect_size'].tolist()
print(top_es)

write_string = 'x\ty\teffect_size\n' + '\n'.join(['\t'.join([str(tsne_x[i]),str(tsne_y[i]),str(top_es[i])]) for i in range(len(tsne_x))])
with open(path.join(EMB_DIR,f'tsne_vis_1k.dat'),'w') as writer:
    writer.write(write_string)

plt.scatter(tsne_df['x'].tolist(),tsne_df['y'].tolist(),c=top_1k['female_effect_size'].tolist())
plt.colorbar()
plt.show()