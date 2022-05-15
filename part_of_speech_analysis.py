import numpy as np
import pandas as pd
from os import path
from flair.data import Sentence
from flair.models import SequenceTagger
from collections import Counter
from math import nan

#GloVe files
TOP_100K_FILE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_100k.csv'
FEMALE_SAVE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_female_upos_analysis.csv'
MALE_SAVE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_male_upos_analysis.csv'
FEMALE_10K_SAVE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_female_10k_upos.csv'
MALE_10K_SAVE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_male_10k_upos.csv'

#fastText files
#TOP_100K_FILE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_100k.csv'
#FEMALE_SAVE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_female_pos_analysis.csv'
#MALE_SAVE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_male_pos_analysis.csv'
#FEMALE_10K_SAVE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_female_10k_pos.csv'
#MALE_10K_SAVE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_male_10k_pos.csv'

#Choose universal or finegrained POS tagger (upos is universal); adjust SAVE file names to match
tagger = SequenceTagger.load("flair/upos-english")
#tagger = SequenceTagger.load("flair/pos-english")

embedding_100k = pd.read_csv(TOP_100K_FILE, na_values=None, keep_default_na=False,names=['word','female_effect_size','p_value'],skiprows=1)

embedding_female = embedding_100k.loc[(embedding_100k['female_effect_size'] >= .5) & (embedding_100k['p_value'] <= .05)]
embedding_top_female = embedding_female.head(10000)
print(embedding_top_female)

embedding_male = embedding_100k.loc[(embedding_100k['female_effect_size'] <= -.5) & (embedding_100k['p_value'] >= .95)]
embedding_top_male = embedding_male.head(10000)
print(embedding_top_male)

#Tag POS of most female and male associated words, then save to file

female_pos_list, male_pos_list = [],[]

for word in embedding_top_female['word'].tolist():
    pred = Sentence(word)
    tagger.predict(pred)
    pos = pred.to_tagged_string().split()[-1]
    female_pos_list.append(pos)

embedding_top_female['pos'] = female_pos_list
embedding_top_female.to_csv(FEMALE_10K_SAVE)

for word in embedding_top_male['word'].tolist():
    pred = Sentence(word)
    tagger.predict(pred)
    pos = pred.to_tagged_string().split()[-1]
    male_pos_list.append(pos)

embedding_top_male['pos'] = male_pos_list
embedding_top_male.to_csv(MALE_10K_SAVE)

#Read saved files back in to validate the write
female_pos_means, female_pos_nums, male_pos_means, male_pos_nums = [],[],[],[]

embedding_top_female = pd.read_csv(FEMALE_10K_SAVE, na_values=None, keep_default_na=False, names=['word','female_effect_size','p_value','pos'],skiprows=1)
female_pos_counts = Counter(embedding_top_female['pos'].tolist())
female_pos_keys = list(female_pos_counts.keys())

embedding_top_male = pd.read_csv(MALE_10K_SAVE, na_values=None, keep_default_na=False, names=['word','female_effect_size','p_value','pos'],skiprows=1)
male_pos_counts = Counter(embedding_top_male['pos'].tolist())
male_pos_keys = list(male_pos_counts.keys())

pos_keys = sorted(list(set(male_pos_keys + female_pos_keys)))

#POS counts by threshold

thresholds = [100,500,1000,2500,5000,10000]

for threshold in thresholds:

    female_sub_df = embedding_top_female.head(threshold)
    male_sub_df = embedding_top_male.head(threshold)

    male_pos_subcounts = Counter(male_sub_df['pos'].tolist())
    female_pos_subcounts = Counter(female_sub_df['pos'].tolist())

    female_sub_means, female_sub_nums, male_sub_means, male_sub_nums = [],[],[],[]

    for k in pos_keys:
        
        if k not in female_pos_subcounts:
            key_mean,key_count = 'N/A',0
        else:
            pos_df = female_sub_df.loc[female_sub_df['pos'] == k]
            key_mean = np.mean(pos_df['female_effect_size'].tolist())
            key_count = female_pos_subcounts[k]
        
        female_sub_nums.append(key_count)
        female_sub_means.append(key_mean)

    female_pos_nums.append(female_sub_nums)
    female_pos_means.append(female_sub_means)

    for k in pos_keys:

        if k not in male_pos_subcounts:
            key_mean,key_count = 'N/A',0
        else:
            pos_df = male_sub_df.loc[male_sub_df['pos'] == k]
            key_mean = np.mean(pos_df['female_effect_size'].tolist())
            key_count = male_pos_subcounts[k]
        
        male_sub_nums.append(key_count)
        male_sub_means.append(key_mean)

    male_pos_nums.append(male_sub_nums)
    male_pos_means.append(male_sub_means)
    print(male_pos_means)

female_pos_data = np.concatenate((np.array([female_pos_nums]).T,np.array([female_pos_means]).T),axis=1).squeeze()
print(female_pos_data)
cols = [f'count_{threshold}' for threshold in thresholds] + [f'mean_{threshold}' for threshold in thresholds]
female_pos_df = pd.DataFrame(female_pos_data,index=pos_keys,columns=cols)

male_pos_data = np.concatenate((np.array([male_pos_nums]).T,np.array([male_pos_means]).T),axis=1).squeeze()
cols = [f'count_{threshold}' for threshold in thresholds] + [f'mean_{threshold}' for threshold in thresholds]
male_pos_df = pd.DataFrame(male_pos_data,index=pos_keys,columns=cols)

female_pos_df.to_csv(FEMALE_SAVE)
male_pos_df.to_csv(MALE_SAVE)