#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gensim import corpora,models,similarities
from collections import defaultdict
from gensim.utils import SaveLoad
from nltk.corpus import stopwords
import re
import numpy as np
from bs4 import BeautifulSoup
import nltk
import pickle


# In[3]:


import pickle
with open('corpus_lsi','rb') as corpus_lsi_file:
    corpus_lsi=pickle.load(corpus_lsi_file)

with open('dictionary','rb') as dictionary_file:
    dictionary=pickle.load(dictionary_file)
    
with open('lsi_model','rb') as lsi_model_file:
    lsi_model=pickle.load(lsi_model_file)


# In[6]:



corpus_title=np.load('titles_improv2.npy')


# In[29]:


#####ENTER YOUR QUERY HERE#####
query=input("Enter your query:")
print(query)


# In[30]:


#query processing
query = query.lower()
query = re.sub(r'[^\w\s]', '', query)
query_bow = dictionary.doc2bow(query.lower().split())

query_lsi = lsi_model[query_bow]
index = similarities.MatrixSimilarity(corpus_lsi)
similarity = index[query_lsi]
similarity = sorted(enumerate(similarity),key=lambda item: -item[1])

#to print the result
answertitle=[]
score=[]
for i in range(len(similarity)):
    answertitle.append(corpus_title[similarity[i][0]])
    score.append(similarity[i])
    
for j in range(10):
    print(answertitle[j],"\n")


# In[ ]:





# In[ ]:




