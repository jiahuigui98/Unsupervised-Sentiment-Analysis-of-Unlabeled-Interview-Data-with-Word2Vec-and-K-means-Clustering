#!/usr/bin/env python
# coding: utf-8

# In[5]:


import nltk # Import the NLTK library
from nltk import word_tokenize,sent_tokenize
from nltk import pos_tag
from nltk import FreqDist # Import the FreqDist function from NLTK
import math # Import math library
from scipy.spatial import distance
from string import punctuation
import re
import nltk
nltk.download('stopwords') # Download the stopwords lists from NLTK
import pandas as pd
from nltk.corpus import stopwords # Import the stop words lists from NLTK
# Print the "stopwords_english" list


# Read the files of interview data

# In[6]:


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


# Add stopwords to the list

# In[7]:


stop_words = stopwords.words('english') # Load the stop words list for English in variable "stopwords_english"

stop_words.append('oh')
stop_words.append('yeah')
stop_words.append('yea')
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
stop_words.append('ooh')
stop_words.append('yes')
stop_words.append('laugh')
stop_words.append('laughs')
stop_words.append('laughter')

print(stop_words) 


# Participants’ name

# In[ ]:


Paritcipants_list=['lm','pl','ju','ja','mi','cl','ra','liz','pa','su','ro','ra','lo','al',
                   'ed','liz','ni','kid1','re','ch','li','tr','x','al','kf','mf','ca','pete',
                   'dad','mom','ch','li','tr','x','al','kf','mf','ca','pete','dad','mum','ni',
                   'kid2','ra','lo','al','ed','r','p','r.','mr.','s.']


# In[8]:


interviewee_list=['lm','pl','ju','ja','mi','cl','ra','liz','pa','su','ro','ra','lo','an','al','ed','liz','ni','kid1','re','ch','li','tr','x','al','kf','mf','ca','pete','dad','mom','re','ch','li','tr','x','al','kf','mf','ca','pete','dad','mum','ni','kid2','ra','lo','an','al','ed','r','p','r.']


# In[9]:


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


# In[10]:


docs=[text_FG1,text_ut1,text_ut2]


# #### Sentence tokenization

# In[11]:


sents=[]
for doc in docs:
    sents.append(sent_tokenize(doc))


# In[12]:


corpus_sents=[]
for doc in sents:
    for i in doc:
        corpus_sents.append(i)


# In[13]:


result_list = [re.split(r'\n',sent) for sent in corpus_sents]#Divided by line break


# In[14]:


result_list2=[]#47872
for sents in result_list:
    for sent in sents:
        result_list2.append(sent)


# In[16]:


result_list3 = [re.split(r':',sent) for sent in result_list2]#Divided by colon


# In[17]:


corpus_sents_complete=[]#47872
for sents in result_list3:
    for sent in sents:
        corpus_sents_complete.append(sent.lower().strip())


# In[19]:


#Removal of elements of length 1
for i in corpus_sents_complete:
    if len(i)==1:
        corpus_sents_complete.remove(i)


# In[20]:


#Removing participant names from the results
for i in corpus_sents_complete:
    if i in Participants_list:
        corpus_sents_complete.remove(i)


# In[21]:


#Removing the empty element
while '' in corpus_sents_complete:
    corpus_sents_complete.remove('')


# In[22]:


#Calculate the number of sentences after the sentence tokenization
print(len(corpus_sents_complete))


# In[23]:


#Storing the results in the form of Dataframe
corpus_sents=pd.DataFrame(corpus_sents_complete)
corpus_sents.columns=["sentence"]


# In[21]:


corpus_sents.to_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/textpre/corpus_sents_oringinal.csv', index=False)


# #### Word Tokenization

# In[24]:


stop_words_filler=['hi','oh','ohh','yeah','yea','okay','ok','um','blah','er',
                  'err','hm','hmm','hmmm','erm','mm','mmm','mmmm','ah','ooh',
                  'uh','huh','uhuh','uhu','uhum','hum','ur','wow','laughter',
                  'laughs','laugh','right','well','yer','coughs','er…','ay','ahh','mhmm']


# In[25]:


tokenized=[word_tokenize(sent.strip()) for sent in corpus_sents_complete]


# In[29]:


#Removing punctuation/numeric characters/participant names
def clean_words_w2v(sent):
    
    #stopwords = [word.strip() for word in stop_words]
    pat = re.compile("[0-9]+")
 
    """ removing numeric characters """
    sent = [token for token in sent if len(pat.findall(''.join(token).strip()))==0]
    """ removing filler words """
    sent = [token for token in sent if token not in stop_words_filler]
 
    """ removing punctuation """
    sent = [token for token in sent if token not in punctuation_list]

    """ removing participant names """
    sent = [token for token in sent if token not in Paritcipants_list]
    """ removing tokens of length 1 """
    sent = [token for token in sent if len(token)!=1]
    
    return sent


# #### Forming the original sentence from the split and cleaned data

# In[30]:


corpus_tokens_cleaned = [clean_words_w2v(sent) for sent in tokenized]#整个corpus


# In[28]:


print(corpus_tokens_cleaned[0:10])


# In[31]:


# if a sentence is only one or two words long,
# the benefit for the training is very small
for i in corpus_tokens_cleaned:
    if len(i)==1:
        corpus_tokens_cleaned.remove(i)


# In[32]:


#Forming sentences
def new_list(ori_list):
    newlist=[]
    for l in ori_list:
        phrase=' '.join(l)
        newlist.append(phrase)
            
    return newlist


# In[33]:


corpus_sents_cleaned=new_list(corpus_tokens_cleaned)


# In[34]:


#removing empty element
while '' in corpus_sents_cleaned:
    corpus_sents_cleaned.remove('')


# In[35]:


print(corpus_sents_cleaned[0:1])


# In[36]:


print(len(corpus_sents_cleaned))


# In[37]:


corpus_sents_cleaned=pd.DataFrame(corpus_sents_cleaned)
corpus_sents_cleaned.columns=["sentence"]


# In[37]:


#corpus_sents_cleaned.to_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/textpre/corpus_sents_cleaned_new.csv', index=False)


# #### Lemmatization

# In[38]:


nltk.download('wordnet') # Download the WordNetLemmatizer package
from nltk.stem import WordNetLemmatizer # Import the WordNetLemmatizer
wnl = WordNetLemmatizer() # Create a WordNetLemmatizer object

nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag


# In[39]:


def penn_to_wordnet(penn_pos_tag):
    tag_dictionary = {'NN':'n', 'JJ':'a','VB':'v', 'RB':'r'}
    try:
        # If the first two characters of the Penn Treebank POS tag are in the ``tag_dictionary''
        return tag_dictionary[penn_pos_tag[:2]]
    except:
        return 'n' # Default to Noun if no mapping available.


# In[40]:


def sentence_stem(sent):
    sent_tagged=pos_tag(sent)
    #for word, tag in sent_tagged:
    sent_tagged=[wnl.lemmatize(word.lower(), pos=penn_to_wordnet(tag)) for word,tag in sent_tagged]#小写

    return sent_tagged


# In[41]:


corpus_tokens_pos=[sentence_stem(sent) for sent in corpus_tokens_cleaned]


# In[42]:


corpus_sents_pos=new_list(corpus_tokens_pos)


# In[43]:


#removing empty element
while '' in corpus_sents_pos:
    corpus_sents_pos.remove('')


# In[44]:


corpus_sents_pos_df=pd.DataFrame(corpus_sents_pos)
corpus_sents_pos_df.columns=["title"]
corpus_sents_pos_df["rate"]=1


# In[45]:


print(corpus_sents_pos_df[:10])


# Storing the results for Word2vec 

# In[46]:


corpus_sents_pos_df.to_csv('/Users/guijiahui/Documents/MDS Data Science/essay/代码/textpre/corpus_sents_pos.csv', index=False)


# 
