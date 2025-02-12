#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install matplotlib')


# In[5]:


get_ipython().system('pip install seaborn')


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("fake_news.csv")
data.head()


# In[12]:


data.shape


# In[7]:


data.info()


# In[8]:


data = data.drop(['id'],axis = 1)


# In[9]:


data.isna().sum()


# In[19]:


data = data.fillna('')


# In[20]:


data['content'] = data['author']+' '+data['title']+" "+data['text']


# In[ ]:


data = data.drop(['title','author','text'],axis = 1)


# In[16]:


data.head()


# In[26]:


data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[27]:


data['content'] = data['content'].str.replace('[^\w\s]','')


# In[28]:


import nltk
nltk.download('stopwords')


# In[29]:


from nltk.corpus import stopwords

stop = stopwords.words('english')

data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[33]:


from nltk.stem import WordNetLemmatizer

from textblob import Word

data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

data['content'].head()


# In[ ]:


x = data[['content']]
y = data['label']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=45, stratify=y)


# In[ ]:


print (X_train.shape)

print (y_train.shape)

print (X_test.shape)

print (y_test.shape)


# In[ ]:


from sklearn.feature_extraction.text import TFid

