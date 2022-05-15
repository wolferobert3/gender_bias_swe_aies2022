import numpy as np
from numpy.core.defchararray import join
from numpy.core.shape_base import _block_slicing
import pandas as pd
from os import path
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns

#Read in embedding

#EMB_DIR = f'D:\\glove_fasttext\\glove.840B.300d'
EMB_DIR = f'D:\\glove_fasttext\\crawl-300d-2M.vec'

#EMB_FILE =f'glove.840B.300d.txt'
EMB_FILE = f'crawl-300d-2M.vec'

#TOP_100K_FILE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_100k.csv'
TOP_100K_FILE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_100k.csv'

#embedding_df = pd.read_csv(path.join(GLOVE_DIR,f'glove.840B.300d.txt'),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE)
embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, skiprows=1, keep_default_na=False,quoting=csv.QUOTE_NONE)

#Get mean cosine similarities with Big Tech words

big_tech_words = ['Google', 'Amazon', 'Facebook', 'Microsoft', 'Apple', 'Nvidia', 'Intel', 'IBM', 'Huawei', 'Samsung', 'Uber', 'Alibaba']

big_tech_embs = embedding_df.loc[[word for word in big_tech if word in embedding_df.index]].to_numpy()
big_tech_normed = big_tech_embs / np.linalg.norm(big_tech_embs,axis=-1,keepdims=True)

all_embs = embedding_df.to_numpy()
all_embs_normed = all_embs / np.linalg.norm(all_embs,axis=-1,keepdims=True)

associations = all_embs_normed @ big_tech_normed.T
means = np.mean(associations,axis=1)

#Write dataframe to file

big_tech_df = pd.DataFrame(means,index=embedding_df.index.tolist(),columns=['big_tech_es'])
largest = big_tech_df.nlargest(10000,'big_tech_es')

#largest.to_csv(path.join(EMB_DIR,f'big_tech_associations_glove.csv'))
largest.to_csv(path.join(EMB_DIR,f'big_tech_associations_ft.csv'))