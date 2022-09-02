#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import jieba
import os
import re
import numpy as np
#import jieba.posseg as psg
import networkx as nx
import pandas as pd
import math


# In[4]:


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


# In[3]:


#Focus Group 1
f = open("/Users/guijiahui/Documents/MDS Data Science/essay/dataset/Virtual consumers/data/Focus Group 1.txt", "r") # Opens the file for reading only ("r")
text_FG1_original = f.read() # Store the contents of the file in variable "text". read() returns all the contents of the file
f.close() # Close the file
#user trial 1
f = open("/Users/guijiahui/Documents/MDS Data Science/essay/dataset/Virtual consumers/data/user trial 1.txt", "r") # Opens the file for reading only ("r")
text_ut1_original = f.read()
f.close() 

#print(text_ut1_2)
#user trial 2
f = open("/Users/guijiahui/Documents/MDS Data Science/essay/dataset/Virtual consumers/data/user trial 2.txt", "r") # Opens the file for reading only ("r")
text_ut2_original = f.read()
f.close() 


# In[4]:


stop_words = stopwords.words('english') # Load the stop words list for English in variable "stopwords_english"

stop_words.append('oh')
stop_words.append('yeah')
stop_words.append('yea')
stop_words.append('okay')
stop_words.append('ok')
stop_words.append('um')
stop_words.append('urm')
stop_words.append('blah')
stop_words.append('er')
stop_words.append('hm')
stop_words.append('erm')
stop_words.append('mmmm')
stop_words.append('erm')
stop_words.append('hmm')
stop_words.append('hmmm')
stop_words.append('ah')
stop_words.append('mmm')
stop_words.append('mm')
stop_words.append('cos')
stop_words.append('ooh')
stop_words.append('yes')
stop_words.append('people')
stop_words.append('things')
stop_words.append('thing')
stop_words.append('something')
stop_words.append('anything')
stop_words.append('laugh')
stop_words.append('laughter')
stop_words.append('lots')
stop_words.append('lot')
stop_words.append('sort')
stop_words.append('kind')
stop_words.append('think')
print(stop_words) 


# In[5]:


import string
punctuation_list=list(string.punctuation)
punctuation_list.append("……")
punctuation_list.append("‘")
punctuation_list.append("’")
punctuation_list.append("…")
punctuation_list.append("–")
punctuation_list.append("”")
punctuation_list.append("“")
punctuation_list.append("...")
punctuation_list.append("....")
punctuation_list.append(".....")
punctuation_list.append("......")
print(punctuation_list)


# In[6]:


Paritcipants_list=['lm','pl','ju','ja','mi','cl','ra','liz','pa','su','ro','ra','lo','al',
                   'ed','liz','ni','kid1','re','ch','li','tr','x','al','kf','mf','ca','pete',
                   'dad','mom','ch','li','tr','x','al','kf','mf','ca','pete','dad','mum','ni',
                   'kid2','ra','lo','al','ed','r','p','r.','mr.','s.']


# In[7]:



text_FG1_r=text_FG1_original.replace('=','')
text_FG1_rr=text_FG1_r.replace('-','')
text_FG1=text_FG1_rr.replace('…','')
text_ut1_r=text_ut1_original.replace('=','')

text_ut1_rr=text_ut1_r.replace('-','')
text_ut1=text_ut1_rr.replace('…','')
text_ut2_r=text_ut2_original.replace('=','')

text_ut2_rr=text_ut2_r.replace('-','')
text_ut2=text_ut2_rr.replace('…','')


# In[8]:


docs=[text_FG1,text_ut1,text_ut2]


# In[9]:


def tokenize_doc(docs):
    """
    :params: docs——多篇文档
    """
    docs_tokenized = []
    for doc in docs:
        doc = re.sub('[\n\r\t]+',' ',doc)
 
        """ 1：sentence tokenization """
        sents = sent_tokenize(doc)
 
        """ 2：word tokenization  """
        sents_tokenized = [word_tokenize(sent.strip()) for sent in sents]
        docs_tokenized.append(sents_tokenized)
 
    return docs_tokenized


# In[49]:


a=tokenize_doc(docs)
print(a)
b=[]
for doc in a:
    for sent in doc:
        b.append(sent)
#print(b)


# In[11]:


def tokenize_sent(docs):
    docs_tokenized=[]
    for doc in docs:
        doc = re.sub('[\n\r\t]+',' ',doc)
        sents = sent_tokenize(doc)
        sents_tokenized = [sent.strip() for sent in sents]
        docs_tokenized.append(sents_tokenized)
        
    return docs_tokenized


# In[12]:


sents=tokenize_sent(docs)


# In[13]:


#去除Stop words/punctuation/标点符号/数字
def clean_words(sent):
    
    #stopwords = [word.strip() for word in stop_words]
    pat = re.compile("[0-9]+")
    
    sent=[token.lower() for token in sent]

    """ removing stopwords """
    sent = [token for token in sent if token not in stop_words]
 
    """ removing numeric character """
    sent = [token for token in sent if len(pat.findall(''.join(token).strip()))==0]    
 
    """ removing punctuation """
    sent = [token for token in sent if token not in punctuation_list]
    """ Words of length 1 in token, mostly name removed """
    sent = [token for token in sent if len(token)!=1]
    """ removing participant names """
    sent = [token for token in sent if token not in participants_list]
    
    return sent


# In[14]:


docs_cleaned = [clean_words(sent) for sent in b]

#print(docs_cleaned)


# In[15]:


doc_pos=[pos_tag(sent) for sent in docs_cleaned ]
print(doc_pos)


# In[16]:


#Retain only nouns

def tag_N(pos_tags):
    tags=["NN","NNS","NNP","NNPS"]
    ret = []
    for word,pos in pos_tags:
            if (pos in tags):
                ret.append(word)
    return ret


# In[19]:


doc_pos_n=[tag_N(sent) for sent in doc_pos]
print(doc_pos_n)


# In[20]:


#Lemmatization
nltk.download('wordnet') # Download the WordNetLemmatizer package
from nltk.stem import WordNetLemmatizer # Import the WordNetLemmatizer
wnl = WordNetLemmatizer() # Create a WordNetLemmatizer object

nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag


# In[21]:


def penn_to_wordnet(penn_pos_tag):
    tag_dictionary = {'NN':'n', 'JJ':'a','VB':'v', 'RB':'r'}
    try:
        # If the first two characters of the Penn Treebank POS tag are in the ``tag_dictionary''
        return tag_dictionary[penn_pos_tag[:2]]
    except:
        return 'n' # Default to Noun if no mapping available.


# In[22]:


def sentence_stem(sent):
    sent_tagged=pos_tag(sent)
    #for word, tag in sent_tagged:
    sent_tagged=[wnl.lemmatize(word.lower(), pos=penn_to_wordnet(tag)) for word,tag in sent_tagged]#小写

    return sent_tagged


# In[23]:


doc_stem_n=[sentence_stem(sent) for sent in doc_pos_n]


# #### Calculate TF-IDF

# In[5]:


from collections import defaultdict
import math
import operator


# In[29]:


def feature_select(list_words):

    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
 
    #Calculate the TF value for each word
    word_tf={}  #Store the tf value of each word
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    #Calculate the IDF value for each word
    doc_num=len(list_words)
    word_idf={} #Store the idf value of each word
    word_doc=defaultdict(int) # Store the number of documents containing the word
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 
    #Calculate the value of TF*IDF for each word
    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
    
    # Sorting the dictionary by value from largest to smallest
    word_tfidf = pd.DataFrame({'word':list(word_tf_idf.keys()),'freq':list(word_tf_idf.values())})
    word_tfidf = word_tfidf.sort_values(by='freq',ascending=False)
    #dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    #dict_feature_list=list(dict(dict_feature_select).keys())
    
    return word_tfidf


# In[30]:


word_tfidf=feature_select(doc_stem_n)


# In[31]:


word_tfidf.to_excel("/Users/guijiahui/Documents/MDS Data Science/essay/代码/word_freq.xlsx",index=False)


# In[32]:


features_n=feature_select(doc_stem_n) 
features=features_n[:100]
print(features)


# #### Wordcloud Visualization

# In[33]:


pip install wordcloud


# In[6]:


import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import os
import random
from imageio import imread

from wordcloud import WordCloud,ImageColorGenerator


# In[20]:


word_freq = pd.read_excel(".../wordtfidf50.xlsx")
word = word_freq.word 
value = word_freq.freq  
dic = dict(zip(word,value))


# In[27]:


d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

def makeImage(text):
    back_coloring = imread(path.join(d, '.../E.jpg'))#picture path

    wc = WordCloud(background_color="white", max_words=1000, mask=back_coloring,
                  contour_width=3, contour_color='steelblue',scale=20,width=400,height=200)
    # generate word cloud
    wc.generate_from_frequencies(text)
    wc.to_file(path.join(d, '.../E_tf-idf.jpeg'))#storage path
    
    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# In[28]:


makeImage(dic)

