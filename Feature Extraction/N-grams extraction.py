#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Focus Group 1
f = open("/Users/guijiahui/Documents/MDS Data Science/essay/dataset/Virtual consumers/data/Focus Group 1.txt", "r") # Opens the file for reading only ("r")
text_FG1 = f.read() # Store the contents of the file in variable "text". read() returns all the contents of the file
f.close() # Close the file
#user trial 1
f = open("/Users/guijiahui/Documents/MDS Data Science/essay/dataset/Virtual consumers/data/user trial 1.txt", "r") # Opens the file for reading only ("r")
text_ut1 = f.read()
f.close() 

#print(text_ut1_2)
#user trial 2
f = open("/Users/guijiahui/Documents/MDS Data Science/essay/dataset/Virtual consumers/data/user trial 2.txt", "r") # Opens the file for reading only ("r")
text_ut2 = f.read()
f.close() 


# In[4]:


import nltk # Import the NLTK library
from nltk import word_tokenize,sent_tokenize
from nltk import FreqDist # Import the FreqDist function from NLTK
import math # Import math library
from scipy.spatial import distance
from string import punctuation

import nltk
nltk.download('stopwords') # Download the stopwords lists from NLTK

from nltk.corpus import stopwords # Import the stop words lists from NLTK
stopwords_english = stopwords.words('english') # Load the stop words list for English in variable "stopwords_english"
print(stopwords_english) # Print the "stopwords_english" list


# In[22]:


# Load the stop words list for English in variable "stopwords_english"
stopwords_english.append('oh')
stopwords_english.append('yeah')
stopwords_english.append('yea')
stopwords_english.append('um')
stopwords_english.append('urm')
stopwords_english.append('blah')
stopwords_english.append('er')
stopwords_english.append('hm')
stopwords_english.append('erm')
stopwords_english.append('mmmm')
stopwords_english.append('erm')
stopwords_english.append('hmm')
stopwords_english.append('hmmm')
stopwords_english.append('ah')
stopwords_english.append('mmm')
stopwords_english.append('mm')
stopwords_english.append('ooh')
stopwords_english.append('yes')
stopwords_english.append('yes')
stopwords_english.append('laugh')
stopwords_english.append('laughs')
stopwords_english.append('laughter')
stopwords_english.append('doo')
stopwords_english.append('ok')
stopwords_english.append('okay')
stopwords_english.append('cos')
stopwords_english.append('beep')
stopwords_english.append('was')
stopwords_english.append('has')

#print(stopwords_english) 


# In[23]:


Participants_list=['lm','pl','ju','ja','mi','cl','ra','liz','pa','su','ro','ra','lo','an','al','ed','liz','ni','kid1','re','ch','li','tr','x','al','kf','mf','ca','pete','dad','mom','re','ch','li','tr','x','al','kf','mf','ca','pete','dad','mum','ni','kid2','ra','lo','an','al','ed','r','p','r.']


# In[9]:




import os,re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from itertools import chain
 
from nltk import pos_tag, word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[10]:


""" step 1：tokenization """
def tokenize_doc(docs):
    """
    :params: docs——Multiple documents
    """
    docs_tokenized = []
    for doc in docs:
        doc = re.sub('[\n\r\t]+',' ',doc)
 
        """ 1：sentence tokenization """
        sents = sent_tokenize(doc)
 
        """ 2：word tokenization """
        sents_tokenized = [word_tokenize(sent.strip()) for sent in sents]
 
    return docs_tokenized


# In[19]:


docs=[text_FG1,text_ut1,text_ut2]
docs_tokenized=tokenize_doc(docs)
print(docs_tokenized)


# In[11]:


""" step2：generate n-gram"""
def gene_ngram(sentence,n=3,m=2):
    """
    ----------
    sentence: tokenized sentence
    n: take 3, then 3-gram
    m: take 1, then keep 1-gram
    ----------
    """
    if len(sentence) < n:
        n = len(sentence)
 
    ngrams = [sentence[i-k:i] for k in range(m, n+1) for i in range(k, len(sentence)+1)]
    return ngrams


# In[12]:


""" if there is a word of length 1 in the n-gram """
def clean_by_len(gram):
 
    for word in gram:
        if len(word) < 2:
            return False
 
    return True
 
 
""" step3：Filter words by stopwords list and pos tags """
def clean_ngrams(ngrams):
    """
    :params: ngrams
    """
    stopwords = [word.strip() for word in stopwords_english]
    participants= [word.strip() for word in Participants_list]
    pat = re.compile("[0-9]+")
 
    """ removing ngram with stopwords_english in it"""
    ngrams = [gram for gram in ngrams if len(set(stopwords).intersection(set(gram)))==0]
    """ removing ngram with participant names in it"""
    ngrams = [gram for gram in ngrams if len(set(participants).intersection(set(gram)))==0]
 
    """ removing ngram with numeric character in it """
    ngrams = [gram for gram in ngrams if len(pat.findall(''.join(gram).strip()))==0]    
 
    """ removing ngram with the token whose length is 1 in it """
    ngrams = [gram for gram in ngrams if clean_by_len(gram)]
 
    """ Only nouns, verbs and adjectives are retained """
    allow_pos_one = ["NN","NNS","NNP","NNPS"]
    allow_pos_two = ["NN","NNS","NNP","NNPS","JJ","JJR","JJS"]
    allow_pos_three = ["NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ","JJ","JJR","JJS"]
 
    ngrams_filter = []
    for gram in ngrams:
        words,pos = zip(*pos_tag(gram))
 
        """ If one word is extracted as a keyword, it must be a noun """
        if len(words) == 1:
            if not pos[0] in allow_pos_one:
                continue   
            ngrams_filter.append(gram)
 
        else:
        """ If the 2-gram/3-gram is extracted then it must begin with a noun, verb or adjective and end with a noun """
            if not (pos[0] in allow_pos_three and pos[-1] in allow_pos_one):
                continue  
            ngrams_filter.append(gram)
 
    return ngrams_filter


# In[14]:


def calcu_tf_idf(documents):

 
    """ Specify vocab, otherwise the n-gram calculation will be wrong """
    vocab = set(chain.from_iterable([doc.split() for doc in documents]))
 
    vec = TfidfVectorizer(vocabulary=vocab)
    D = vec.fit_transform(documents)
    voc = dict((i, w) for w, i in vec.vocabulary_.items())
 
    features = {}
    for i in range(D.shape[0]):
        Di = D.getrow(i)
        features[i] = list(zip([voc[j] for j in Di.indices], Di.data))
 
    return features


# In[15]:


def get_ngram_keywords(docs_tokenized,topk=5,n_=2):
 
    """ 1：n-gram """
    docs_ngrams = [gene_ngram(doc,n=n_,m=n_) for doc in docs_tokenized]
 
    """ 2: filter the ngrams """
    docs_ngrams = [clean_ngrams(doc) for doc in docs_ngrams]
 
    docs_ = []
    for doc in docs_ngrams:
        docs_.append(' '.join(['_'.join(ngram) for ngram in doc]))
 
    """ 3: calculate tf-idf，extract keywords """
    features = calcu_tf_idf(docs_)
 
    docs_keys = []
    for i,pair in features.items():
        topk_idx = np.argsort([v for w,v in pair])[::-1][:topk]
        docs_keys.append([pair[idx][0] for idx in topk_idx])
 
    return [[' '.join(words.split('_')) for words in doc ]for doc in docs_keys] 
 
 
""" step 5：keyword extraction """
def get_keywords(docs_tokenized,topk):
 
    """ 1: tokenization """  
    docs_tokenized = [list(chain.from_iterable(doc)) for doc in docs_tokenized]
    """ 2: keyword extraction，including unigram，bigram and trigram """
    docs_keys = []
    for n in [1,2,3]:
        if n == 1:
            #""" 3: unigram, a morphological reduction is also required """
            docs_tokenized = [[lemmatizer.lemmatize(word) for word in doc] for doc in docs_tokenized]
        keys_ngram = get_ngram_keywords(docs_tokenized, topk,n_=n)
        docs_keys.append(keys_ngram)
 

    #return [uni+bi+tri for uni,bi,tri in docs_keys]
    return [uni+bi+tri for uni,bi,tri in zip(*docs_keys)]


# In[30]:


ngram_extraction=get_keywords(docs_tokenized,40)


# In[31]:


print(ngram_extraction)


# In[27]:


import pandas as pd


# In[28]:



ngram_extraction_df=pd.DataFrame(ngram_extraction)
    


# In[29]:


ngram_extraction_df.to_csv('/Users/guijiahui/Documents/MDS Data Science/essay/Feature/TF-IDF.csv', index=False)


# In[ ]:





# In[ ]:




