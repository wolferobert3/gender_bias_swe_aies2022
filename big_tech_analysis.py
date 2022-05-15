import numpy as np
from numpy.core.defchararray import join
from numpy.core.shape_base import _block_slicing
import pandas as pd
from os import path
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns

def SC_WEAT(w, A, B, permutations):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T
    joint_associations = np.concatenate((A_associations,B_associations),axis=-1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations,ddof=1)

    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:,:midpoint],axis=1) - np.mean(sample_distribution[:,midpoint:],axis=1)
    p_value = 1 - norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1))

    return effect_size, p_value

#EMB_DIR = f'D:\\glove_fasttext\\glove.840B.300d'
EMB_DIR = f'D:\\glove_fasttext\\crawl-300d-2M.vec'

#EMB_FILE =f'glove.840B.300d.txt'
EMB_FILE = f'crawl-300d-2M.vec'

#TOP_100K_FILE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_100k.csv'
TOP_100K_FILE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_100k.csv'

#Read in embeddings, skip row for FT
#embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE)
embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, skiprows=1, keep_default_na=False,quoting=csv.QUOTE_NONE)

#Get gender stimuli
female_stimuli = ['female','woman','girl','sister','she','her','hers','daughter']
male_stimuli = ['male','man','boy','brother','he','him','his','son']

female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()

#Read in big tech associations and take the words that are associated with big tech in both embeddings
largest_ft = pd.read_csv(path.join(EMB_DIR,f'big_tech_associations_ft.csv'),index_col=0)
largest_glove = pd.read_csv(path.join(GLOVE_DIR,f'big_tech_associations_glove.csv'),index_col=0)
joint = [i for i in largest_ft.index.tolist() if i in largest_glove.index]

#Get gender associations of big tech words
joint_vals = []

for word in joint:
    joint_emb = embedding_df.loc[word].to_numpy()
    es, p = SC_WEAT(joint_emb,female_embeddings,male_embeddings,1000)
    joint_vals.append([es,p])

joint_arr = np.array(joint_vals)
big_tech = pd.DataFrame(joint_arr,index=joint,columns=['Effect_Size','P_Value'])
big_tech.to_csv(path.join(EMB_DIR,f'big_tech_weats.csv'))

#Write big tech words to file
words = big_tech.index.tolist()
with open(path.join(GLOVE_DIR,f'big_tech_words.txt'),'w') as writer:
    writer.write(', '.join(sorted(words,key=str.lower)))

#Get percentage of big tech words with minimum gender effect size

es_list = [0,.2,.5,.8]

pct_female, pct_male = [],[]

for es in es_list:
    print(es)
    female_df = big_tech[(big_tech.Effect_Size >= es)]
    pct_female.append(len(female_df.index.tolist())/len(big_tech.index.tolist())

    male_df = big_tech[(big_tech.Effect_Size <= -es)]
    pct_male.append(len(male_df.index.tolist())/len(big_tech.index.tolist())

print(pct_female)
print(pct_male)