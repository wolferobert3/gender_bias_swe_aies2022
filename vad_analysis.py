import numpy as np
from numpy.core.numeric import full
import pandas as pd
from os import path
import csv
from scipy.stats.stats import spearmanr,pearsonr
from wordfreq import word_frequency, zipf_frequency

SOURCE_DIR = f'D:\\glove_fasttext\\glove.840B.300d'
#SOURCE_DIR = f'D:\\glove_fasttext\\crawl-300d-2M.vec'

embedding_vad = pd.read_csv(f'D:\\glove_fasttext\\glove.840B.300d\\glove_vad_words.csv', quoting=csv.QUOTE_NONE, na_values=None, keep_default_na=False,index_col=0)
#embedding_vad = pd.read_csv(f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_vad_words_p1k.csv', quoting=csv.QUOTE_NONE, na_values=None, keep_default_na=False,index_col=0)

vad_df = pd.read_table(f'D:\\NRC-VAD-Lexicon.txt',sep='\t',quoting=csv.QUOTE_NONE,index_col=0, na_values=None, keep_default_na=False)
base_vad_words = vad_df.index.tolist()

vad_words = [word for word in base_vad_words if word in embedding_vad.index]

val_, dom_, aro_, freq_list, effect_size, p_value = [],[],[],[],[],[]

#Get VAD properties, effect size data only for words in the embeddings
for word in vad_words:

    vad_stat = vad_df.loc[word]
    val_.append(vad_stat['Valence'])
    dom_.append(vad_stat['Dominance'])
    aro_.append(vad_stat['Arousal'])

    freq_list.append(zipf_frequency(word,'en'))

    glove_stat = embedding_vad.loc[word]
    effect_size.append(glove_stat['female_effect_size'])
    p_value.append(glove_stat['female_p_value'])

data = np.array([val_,dom_,aro_,freq_list,effect_size,p_value]).T
cols = ['Valence','Dominance','Arousal','Frequency','Effect_Size','P_Value']

full_df = pd.DataFrame(data,index=vad_words,columns=cols)
full_df.to_csv(path.join(SOURCE_DIR,f'vad_master.csv'))

top_n_freqs = [100,1000,10000,len(vad_words)]

vad = ['Valence','Dominance','Arousal']
vad_comp = ['Dominance','Arousal','Valence']

spearmans, pearsons = [],[]

#Correlations by frequency range

for n in top_n_freqs:

    sub_df = full_df.nlargest(n,'Frequency')

    spearman_es = [spearmanr(sub_df['Effect_Size'],sub_df[signal]) for signal in vad]
    spearman_vad = [spearmanr(sub_df[vad[i]],sub_df[vad_comp[i]]) for i in range(len(vad))]

    spearmans.append(spearman_es+spearman_vad)

cols = ['Valence_ES','Dominance_ES','Arousal_ES','Val_Dom','Dom_Aro','Aro_Val']

vad_spearman_df = pd.DataFrame(spearmans,index=top_n_freqs,columns=cols)
vad_spearman_df.to_csv(path.join(SOURCE_DIR,'vad_frequency_spearman.csv'))

#Correlations by effect size range

spearmans,pearsons=[],[]
effect_size_ranges = [0,.2,.5,.8]

for es in effect_size_ranges:

    sub_df = full_df.loc[full_df['Effect_Size'] >= es]

    spearman_es = [spearmanr(sub_df['Effect_Size'],sub_df[signal]) for signal in vad]
    spearman_vad = [spearmanr(sub_df[vad[i]],sub_df[vad_comp[i]]) for i in range(len(vad))]

    spearmans.append(spearman_es+spearman_vad)

cols = ['Valence_ES','Dominance_ES','Arousal_ES','Val_Dom','Dom_Aro','Aro_Val']

vad_spearman_df = pd.DataFrame(spearmans,index=effect_size_ranges,columns=cols)
vad_spearman_df.to_csv(path.join(SOURCE_DIR,'vad_effect_size_spearman.csv'))

print(vad_spearman_df)