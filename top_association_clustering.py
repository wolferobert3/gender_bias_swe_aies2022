import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
import random
from os import path
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score

EMB_DIR = f'D:\\glove_fasttext\\glove.840B.300d'
#EMB_DIR = f'D:\\glove_fasttext\\crawl-300d-2M.vec'

EMB_FILE =f'glove.840B.300d.txt'
#EMB_FILE = f'crawl-300d-2M.vec'

TOP_100K_FILE = f'D:\\glove_fasttext\\glove.840B.300d\\glove_100k.csv'
#TOP_100K_FILE = f'D:\\glove_fasttext\\crawl-300d-2M.vec\\ft_100k.csv'

#Read in gender association file and get most associated female and male words
embedding_100k = pd.read_csv(TOP_100K_FILE, na_values=None, keep_default_na=False,names=['word','female_effect_size','p_value'],skiprows=1)

embedding_female = embedding_100k.loc[(embedding_100k['female_effect_size'] >= .5) & (embedding_100k['p_value'] <= .05)]
embedding_top_female = embedding_female.head(1000)
top_female_words = embedding_top_female['word'].tolist()

embedding_male = embedding_100k.loc[(embedding_100k['female_effect_size'] <= -.5) & (embedding_100k['p_value'] >= .95)]
embedding_top_male = embedding_male.head(1000)
top_male_words = embedding_top_male['word'].tolist()

#Read in embeddings, skip row only for FT
embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=100000)
#embedding_df = pd.read_csv(path.join(EMB_DIR,EMB_FILE),sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=100000,skiprows=1)

embeddings_female = embedding_df.loc[top_female_words]
embeddings_male = embedding_df.loc[top_male_words]

target_data_female = embeddings_female.to_numpy()
target_data_male = embeddings_male.to_numpy()

#Use elbow method to assess stopping point for female and male clusters

INIT = 3
ITERS = 26

wcss = []

for i in range(INIT, ITERS): 
    kmeans = KMeans(n_clusters=i, random_state=0,algorithm='elkan',init='k-means++',max_iter=1000,n_init=100)
    kmeans.fit(target_data_female)
    wcss.append(kmeans.inertia_)

plt.plot([i for i in range(INIT,ITERS)],ss_)
plt.xticks([i for i in range(INIT,ITERS)])
plt.show()

wcss = []

for i in range(INIT, ITERS): 
    kmeans = KMeans(n_clusters=i, random_state=0,algorithm='elkan',init='k-means++',max_iter=1000,n_init=100)
    kmeans.fit(target_data_male)
    wcss.append(kmeans.inertia_)

plt.plot([i for i in range(INIT,ITERS)],ss_)
plt.xticks([i for i in range(INIT,ITERS)])
plt.show()

#K-Means clustering and transformed coordinates
NUM_CLUSTERS = 11
kmeans_female = KMeans(n_clusters=NUM_CLUSTERS, random_state=0,algorithm='elkan',init='k-means++',max_iter=1000,n_init=100).fit(target_data_female)
kmeans_female_transform = KMeans(n_clusters=NUM_CLUSTERS, random_state=0,algorithm='elkan',init='k-means++',max_iter=1000,n_init=100).fit_transform(target_data_female)

kmeans_male = KMeans(n_clusters=NUM_CLUSTERS, random_state=0,algorithm='elkan',init='k-means++',max_iter=1000,n_init=100).fit(target_data_male)
kmeans_male_transform = KMeans(n_clusters=NUM_CLUSTERS, random_state=0,algorithm='elkan',init='k-means++',max_iter=1000,n_init=100).fit_transform(target_data_male)

#T-SNE coordinates
reduced_dims_female = TSNE().fit_transform(kmeans_female_transform.squeeze())
tsne_df_female = pd.DataFrame(reduced_dims_female,index=top_female_words,columns=['x','y'])
tsne_df_female['word'] = top_female_words
tsne_df_female['cluster'] = kmeans_female.labels_
tsne_df_female.to_csv(path.join(EMB_DIR,f'tsne_clusters_female_1k_{NUM_CLUSTERS}.csv'))

tsne_female_x = tsne_df_female['x'].tolist()
tsne_female_y = tsne_df_female['y'].tolist()

reduced_dims_male = TSNE().fit_transform(kmeans_male_transform.squeeze())
tsne_df_male = pd.DataFrame(reduced_dims_male,index=top_male_words,columns=['x','y'])
tsne_df_male['word'] = top_male_words
tsne_df_male['cluster'] = kmeans_male.labels_
tsne_df_male.to_csv(path.join(EMB_DIR,f'tsne_clusters_male_1k_{NUM_CLUSTERS}.csv'))

tsne_male_x = tsne_df_male['x'].tolist()
tsne_male_y = tsne_df_male['y'].tolist()

#Write cluster coordinates to .dat file
write_string = 'x\ty\tcluster\tword\n' + '\n'.join(['\t'.join([str(tsne_female_x[i]),str(tsne_female_y[i]),str(kmeans_female.labels_[i]),str(top_female_words[i])]) for i in range(len(tsne_female_x))])
with open(path.join(EMB_DIR,f'tsne_clusters_female_male_vis_elkan_{NUM_CLUSTERS}.dat'),'w',encoding='utf8') as writer:
    writer.write(write_string)

write_string = 'x\ty\tcluster\tword\n' + '\n'.join(['\t'.join([str(tsne_male_x[i]),str(tsne_male_y[i]),str(kmeans_male.labels_[i]),str(top_male_words[i])]) for i in range(len(tsne_male_x))])
with open(path.join(EMB_DIR,f'tsne_clusters_male_male_vis_elkan_{NUM_CLUSTERS}.dat'),'w',encoding='utf8') as writer:
    writer.write(write_string)

female_write_string, male_write_string = '',''

#Write female and male clustered words to text files
for i in range(NUM_CLUSTERS):
    cluster_df_female = tsne_df_female.loc[tsne_df_female['cluster'] == i]
    cluster_df_male = tsne_df_male.loc[tsne_df_male['cluster'] == i]

    female_write_string += f'Cluster {i}:' + ', '.join(sorted(cluster_df_female.index.tolist(),key=str.lower)) + '\n'
    male_write_string += f'Cluster {i}:' + ', '.join(sorted(cluster_df_male.index.tolist(),key=str.lower)) + '\n'

with open(path.join(EMB_DIR,f'female_clusters_{NUM_CLUSTERS}.txt'),'w',encoding='utf8') as writer:
    writer.write(female_write_string)

with open(path.join(EMB_DIR,f'male_clusters_{NUM_CLUSTERS}.txt'),'w',encoding='utf8') as writer:
    writer.write(male_write_string)