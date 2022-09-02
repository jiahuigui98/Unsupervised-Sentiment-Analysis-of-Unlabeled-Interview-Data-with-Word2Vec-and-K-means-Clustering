#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# In[4]:


from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn 


# In[127]:


##Enter the file path of the corpus
final_file = pd.read_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/textpre/corpus_sents_pos.csv')


# In[128]:


#Enter the file path of the sentiment dicitonary built with CBOW and Skip-gram
sentiment_map = pd.read_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/sentiment_analysis/sentiment_dictionary.csv')
sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coeff.values))


# Getting tfidf scores of words in every sentence, and replacing them with their associated tfidf weights:

# In[129]:


file_weighting = final_file.copy()


# calculate TF-IDF values of the words in each sentence

# In[131]:


tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
tfidf.fit(file_weighting.title)
features = pd.Series(tfidf.get_feature_names())
transformed = tfidf.transform(file_weighting.title)


# Replacing words in sentences with their tfidf values

# In[132]:


def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer

    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    return list(map(lambda y:dictionary[f'{y}'], x.title.split()))


# In[133]:


get_ipython().run_cell_magic('time', '', 'replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)')


# Replacing words in sentences with their sentiment scores

# In[134]:


def replace_sentiment_words(word, sentiment_dict):
    '''
    replacing each word with its associated sentiment score from sentiment dict
    '''
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out


# In[135]:


replaced_closeness_scores = file_weighting.title.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))


# Merging both previous steps and getting the predictions:

# In[136]:


replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.title, file_weighting.rate]).T
replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'rate']
replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
replacement_df['prediction'] = (replacement_df.sentiment_rate>0).astype('int8')
replacement_df['rate'] = [1 if i==1 else 0 for i in replacement_df.rate]


# In[137]:


print(replacement_df[0:40])


# Storing the results of sentiment analysis

# In[ ]:


replacement_df.to_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/sentiment_analysis/word2vec_results.csv', index=False)


# #### Comparing the results of sentiment analysis with the results of manual annotation

# Read manually labeled data into python

# In[167]:


manual_label = pd.read_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/sentiment_analysis/manual_label.csv')


# In[168]:


replacement_df["manual_label"]=pd.DataFrame(data=manual_label['label'])


# In[169]:


w2v_label=replacement_df[12271:12979]#input the number of rows with manual label


# In[171]:


#


# In[187]:


predicted_classes = w2v_label.prediction
y_test = w2v_label.manual_label

conf_matrix_1 = pd.DataFrame(confusion_matrix(w2v_label.manual_label, w2v_label.prediction))
print('Confusion Matrix')
display(conf_matrix)
conf_matrix.columns = ['Predicted Neg Sentences','Predicted Pos Sentences']
conf_matrix.index = ['Actual Neg Sentences', 'Actual Pos Sentences']
sn.heatmap(conf_matrix,annot=True,fmt="d",cmap="RdBu_r")

test_scores = accuracy_score(y_test,predicted_classes), precision_score(y_test, predicted_classes), recall_score(y_test, predicted_classes), f1_score(y_test, predicted_classes)

print('\n \n Scores')
scores = pd.DataFrame(data=[test_scores])
scores.columns = ['accuracy', 'precision', 'recall', 'f1']
scores = scores.T
scores.columns = ['scores']
display(scores)


# #### 和textblob比

# In[190]:


predicted_classes = w2v_label.prediction
y_test = w2v_label.senti_textblob

conf_matrix_2 = pd.DataFrame(confusion_matrix(w2v_label.senti_textblob, w2v_label.prediction))
print('Confusion Matrix')
#display(conf_matrix_2)
conf_matrix_2.columns = ['Predicted Neg Sentences','Predicted Pos Sentences']
conf_matrix_2.index = ['Actual Neg Sentences', 'Actual Pos Sentences']
sn.heatmap(conf_matrix_2,annot=True,fmt="d",cmap="RdBu_r")

test_scores = accuracy_score(y_test,predicted_classes), precision_score(y_test, predicted_classes), recall_score(y_test, predicted_classes), f1_score(y_test, predicted_classes)

print('\n \n Scores')
scores = pd.DataFrame(data=[test_scores])
scores.columns = ['accuracy', 'precision', 'recall', 'f1']
scores = scores.T
scores.columns = ['scores']
display(scores)


# In[ ]:




