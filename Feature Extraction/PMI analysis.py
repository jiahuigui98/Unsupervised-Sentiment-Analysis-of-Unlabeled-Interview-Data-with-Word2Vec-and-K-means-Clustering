#!/usr/bin/env python
# coding: utf-8

# In[3]:


import collections
from collections import Counter
import math
from nltk.util import ngrams
import pandas as pd


# In[11]:


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


# In[14]:


stop_words = stopwords.words('english')


# Read text pre-processed files

# In[5]:


corpus_sw = pd.read_csv('.../corpus_sents_pos.csv')


# In[8]:


corpus_sentence=corpus_sw['title'].tolist()


# Tokenization

# In[12]:


corpus_tokens=[]
for i in corpus_sentence:
    a=word_tokenize(i)
    corpus_tokens.append(a)


# In[16]:


corpus_sw=[]
for sent in corpus_tokens:
    for i in sent:
        corpus_sw.append(i)


# In[23]:


corpus=[]
for i in corpus_sw:
    if i not in stop_words:
        corpus.append(i)


# In[24]:


#print(corpus)


# PMI Analysis

# In[25]:


word_cntdict=collections.Counter(corpus)


# In[26]:


ngram_cntdict=collections.Counter(ngrams(corpus,3))


# In[27]:


tot_freg=sum([word_cntdict[key]for key in word_cntdict])
tot_ng_freg=sum([ngram_cntdict[key]for key in ngram_cntdict])


# In[28]:


words_prob={x:word_cntdict[x]/tot_freg for x in corpus}
print(words_prob)


# In[ ]:


#joy


# In[31]:


keyword="happy"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[39]:


keyword="enjoy"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="pleased"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="interesting"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# Sadness

# In[ ]:


keyword="sad"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="impatient"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="disappointed"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# Anger

# In[ ]:


keyword="annoy"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="brother"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="trouble"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# Fear

# In[ ]:


keyword="fear"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="afraid"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="reluctance"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[37]:


keyword="concern"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[40]:


keyword="concerned"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="scared"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# Disgust

# In[ ]:


keyword="resent"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# In[ ]:


keyword="hate"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))


# Surprise

# In[32]:


keyword="surprise"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(30))


# In[38]:


keyword="surprised"
pair_prob={}
keyword_pmi={}
for word in corpus:
    if word!=keyword:
        pair_prob[(keyword,word)]=sum([ngram_cntdict[keys]for keys in ngram_cntdict if keyword in keys and word in keys])/tot_ng_freg
        if pair_prob[(keyword,word)]==0:
            keyword_pmi[(keyword,word)]=0
        else:
            keyword_pmi[(keyword,word)]=round(math.log((pair_prob[(keyword,word)]/(words_prob[keyword]*words_prob[word])),2),4)
print(Counter(keyword_pmi).most_common(15))

