#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd


# In[8]:


import nltk # Import the NLTK library
from nltk import word_tokenize,sent_tokenize
from nltk import pos_tag
from nltk import FreqDist # Import the FreqDist function from NLTK
import math # Import math library
from scipy.spatial import distance
from string import punctuation

import nltk
nltk.download('stopwords') # Download the stopwords lists from NLTK

from nltk.corpus import stopwords # Import the stop words lists from NLTK
# Print the "stopwords_english" list


# Read text pre-processed files

# In[5]:


corpus_sw = pd.read_csv('.../corpus_sents_pos.csv')


# In[6]:


corpus_sentence=corpus_sw['title'].tolist()


# In[9]:


corpus_tokens=[]
for i in corpus_sentence:
    a=word_tokenize(i)
    corpus_tokens.append(a)


# In[11]:


w2v_model_skip = Word2Vec(corpus_tokens,vector_size = 300, window = 4 , min_count = 3, epochs=200, negative=10,sg=1)


# PCA Visualization

# In[12]:


# Projection of word embeddings onto a two-dimensional space
word_vectors = []
word2ind = {}
for i, w in enumerate(w2v_model_skip.wv.index_to_key): 
    word_vectors.append(w2v_model_skip.wv[w]) 
    word2ind[w] = i 
rawWordVec = np.array(word_vectors)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec) 


# In[69]:


# 绘制星空图
# 绘制所有单词向量的二维空间投影
fig = plt.figure(figsize = (15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'black')

# 绘制几个特殊单词的向量
words = ['concerned','concern','happy','pleased','sad','impatient','site','website','online','shopping','e-commerce','commerce','computer','pc','telephone','phone']

# 设置中文字体 否则乱码
#zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf', size=16)
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'orange',markersize=10)
        plt.text(xy[0], xy[1], w, alpha = 1, color = 'red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Skip-gram model:PCA visualization')


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# t-SNE Visualization

# In[70]:


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components= 2).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('Skip-gram model:t-SNE visualization for {}'.format(word.title()))


# In[71]:


tsnescatterplot(w2v_model_skip, 'e-commerce', ['love', 'sad', 'mad', 'fear', 'surprise', 'hate','local','online','shopping'])


# In[72]:


tsnescatterplot(w2v_model_skip, 'e-commerce', [i[0] for i in w2v_model_skip.wv.most_similar(negative=["e-commerce"])])


# K-means clustering

# In[17]:



from sklearn.cluster import KMeans
import numpy as np


# In[18]:


model_kmeans = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors)


# In[19]:


from mpl_toolkits import mplot3d


# In[20]:


labels = model_kmeans.labels_
print(labels)


# In[22]:


#plt.rcParams['font.sans-serif'] = ['SimHei']         # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 显示负号
fig = plt.figure()
ax = mplot3d.Axes3D(fig)                             # 创建3d坐标轴
colors = ['red', 'blue']
# 绘制散点图 词语 词向量 类标(颜色)
for word, vector, label in zip(w2v_model_skip.wv.index_to_key, word_vectors, labels):
    ax.scatter(vector[0], vector[1], vector[2], c=colors[label], s=100, alpha=0.4)
    ax.text(vector[0], vector[1], vector[2], word, ha='center', va='center')
plt.show()


# Identify which cluster has more positive words

# In[32]:


w2v_model_skip.wv.similar_by_vector(model_kmeans.cluster_centers_[0], topn=100, restrict_vocab=None)


# In[23]:


negative_cluster_center = model_kmeans.cluster_centers_[0]
positive_cluster_center = model_kmeans.cluster_centers_[1]


# In[43]:


w2v_model_skip.wv.similarity('dislike','upset') #积极


# In[42]:


w2v_model_skip.wv.similarity('enjoyable','upset') #萧极


# #### Build Sentiment Lexcion

# In[44]:


word_vectors_all=w2v_model_skip.wv


# In[45]:


positive_cluster_index = 1
positive_cluster_center = model_kmeans.cluster_centers_[positive_cluster_index]
negative_cluster_center = model_kmeans.cluster_centers_[1-positive_cluster_index]


# In[75]:


words = pd.DataFrame(word_vectors_all.index_to_key)
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: word_vectors_all[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: model_kmeans.predict([np.array(x)]))
words.cluster = words.cluster.apply(lambda x: x[0])


# calculate the sentiment intensity

# In[89]:


words['cluster_value'] = [1 if i==positive_cluster_index else -1 for i in words.cluster]#0和1转换为1和-1
words['closeness_score'] = words.apply(lambda x: 1/(model_kmeans.transform([x.vectors]).min()), axis=1)
words['sentiment_coeff'] = words.closeness_score * words.cluster_value


# In[48]:


words[['words', 'sentiment_coeff']].to_csv('.../sentiment_dictionary_skip.csv', index=False)

