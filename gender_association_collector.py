import numpy as np
import pandas as pd
from scipy.stats import norm
from os import path

#SC-WEAT function
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

#This code obtains the SC-WEAT association for the top 100,000 most frequent words in each embedding, and for the words in NRC-VAD Lexicon

#Constants
STEP = 10000
PERMUTATIONS = 10000
GLOVE_DIR = f'D:\\glove_fasttext\\glove.840B.300d'
FT_DIR = f'D:\\glove_fasttext\\crawl-300d-2M.vec'

#Attribute Words
female_stimuli = ['female','woman','girl','sister','she','her','hers','daughter']
male_stimuli = ['male','man','boy','brother','he','him','his','son']

#Skip first row when reading FT, not when reading GloVe
embedding_df = pd.read_csv(path.join(GLOVE_DIR,f'glove.840B.300d.txt'),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False)
print('GloVe loaded')

female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()
embedding_targets = embedding_df.index.tolist()[:100000]

#NRC-VAD Dataframe
vad_df = pd.read_table(f'D:\\NRC-VAD-Lexicon.txt',sep='\t',index_col=0, na_values=None, keep_default_na=False)
vad_words = vad_df.index.tolist()
vad_words = [word for word in vad_words if word in embedding_df.index]

gender_biases, p_values = [],[]

#VAD WEATs - GloVe embedding
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in [str(i) for i in vad_words]])
bias_df = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
bias_df.to_csv(path.join(GLOVE_DIR,f'glove_vad_words.csv'))
print('GloVe VAD')

#10k WEATS at a time - 100k most frequent words - GloVe embedding
for i in range(10):
    targets = embedding_targets[i*STEP:(i+1)*STEP]
    bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in targets])
    bias_df = pd.DataFrame(bias_array,index=targets,columns=['female_effect_size','female_p_value'])
    bias_df.to_csv(path.join(GLOVE_DIR,f'glove_100k_{i}.csv'))

print('GloVe 100k')

#Read in FastText embedding
embedding_ft = pd.read_csv(path.join(FT_DIR,f'crawl-300d-2M.vec'),sep=' ',header=None,skiprows=1,index_col=0, na_values=None, keep_default_na=False)
print('FT loaded')

female_embeddings, male_embeddings = embedding_ft.loc[female_stimuli].to_numpy(), embedding_ft.loc[male_stimuli].to_numpy()
embedding_targets = embedding_ft.index.tolist()[:100000]

#Only VAD words in embedding
vad_words = vad_df.index.tolist()
vad_words = [word for word in vad_words if word in embedding_ft.index]

bias_array = np.array([SC_WEAT(embedding_ft.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in [str(i) for i in vad_words]])
bias_ft = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
bias_ft.to_csv(path.join(FT_DIR,f'ft_vad_words.csv'))
print('FT VAD')

#10k SC-WEATs at a time - 100k most frequent words - FT embedding
for i in range(10):
    targets = embedding_targets[i*STEP:(i+1)*STEP]
    bias_array = np.array([SC_WEAT(embedding_ft.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in targets])
    bias_ft = pd.DataFrame(bias_array,index=targets,columns=['female_effect_size','female_p_value'])
    bias_ft.to_csv(path.join(FT_DIR,f'ft_100k_{i}.csv'))

print('FT 100k')

#Concatenate and save 10k-word association dataframes
#GloVe

concat_ = []
for i in range(10):
    df = pd.read_csv(path.join(GLOVE_DIR,f'glove_100k_{i}_p1k.csv'),names=['word','female_effect_size','p_value'],skiprows=1,index_col='word', na_values=None, keep_default_na=False)
    concat_.append(df)

full_df = pd.concat(concat_,axis=0)
full_df.to_csv(path.join(GLOVE_DIR,f'glove_100k.csv'))

#FastText
concat_ = []
for i in range(10):
    df = pd.read_csv(path.join(FT_DIR,f'ft_100k_{i}_p1k.csv'),names=['word','female_effect_size','p_value'],skiprows=1,index_col='word', na_values=None, keep_default_na=False)
    concat_.append(df)

full_df = pd.concat(concat_,axis=0)
full_df.to_csv(path.join(FT_DIR,f'ft_100k.csv'))