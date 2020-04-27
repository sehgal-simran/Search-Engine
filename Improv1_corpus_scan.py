#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
from collections import Counter
from nltk.util import ngrams
import re 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#loading the file
f = open("wiki_01txt.txt", "r", encoding="utf8")


# In[4]:


#extracting the corpus
soup = BeautifulSoup(f, 'html.parser')
wiki_text = soup.get_text()


# In[5]:


#extracting the documents from the corpus and storing documents and titles
corpus_text=[]
corpus_title=[]
for item in soup.find_all('doc'):
    item_text = item.get_text()
    item_title=item.get('title')
    item_text = item_text.lower()
    item_text = re.sub(r'[^\w\s]', '', item_text)
    unigram = nltk.word_tokenize(item_text)
    #print(unigram)
    #lemmatizing the unigrams
    lemmer=WordNetLemmatizer()
    lemWords_unigram_verb=[lemmer.lemmatize(word,pos="v") for word in unigram]
    lemWords_unigram_adverb=[lemmer.lemmatize(word,pos="a") for word in lemWords_unigram_verb]
    item_text=' '.join([lemmer.lemmatize(word, pos="n") for word in lemWords_unigram_adverb])
    #print(new_corpus)
    corpus_text.append(item_text)
    corpus_title.append(item_title)     
print(corpus_text[0])
print(corpus_text[1])


# In[6]:


#building incidence matrix and extracting vocabulary
corpus = corpus_text
title=corpus_title
vectorizer = CountVectorizer()
tf_raw = vectorizer.fit_transform(corpus)
features=vectorizer.get_feature_names()
print(features)
print(tf_raw.toarray())
print(vectorizer.vocabulary_)


# In[7]:


#building posting list and normalization values list
N=len(corpus)
norm=[0]*N
tf_raw_arr=tf_raw.toarray()
ct=0;
posting_list = {i: [] for i in vectorizer.vocabulary_} 
#print(listKeys)
for j in posting_list:
    ct=0
    for i in tf_raw_arr:
        if i[vectorizer.vocabulary_[j]]>0:
            posting_list[j].append((ct,i[vectorizer.vocabulary_[j]]))
            norm[ct]=norm[ct]+(1+math.log10(i[vectorizer.vocabulary_[j]]))**2
        ct=ct+1
print(posting_list)


# In[8]:


#saving posting list and normalization values list
np.save('posting_list_improv1.npy', posting_list) 
np.save('norm_improv1.npy', norm)
np.save('titles_improv1.npy',title)


# In[9]:


print(len(title))


# In[ ]:





# In[ ]:




