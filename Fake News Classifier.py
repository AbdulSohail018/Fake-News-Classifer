#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


df=pd.read_csv('C:/Users/Sohail/Desktop/fakenewscheck.csv')


# In[3]:


df.head(10)


# In[4]:


labelling=df.label
labelling.head()


# ### 1.  Using TfidVectorizer and Passive Aggressive Classifier

# In[5]:


x_train,x_test,y_train,y_test=train_test_split(df['text'], labelling, test_size=0.3, random_state=10)


# In[6]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.6)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[8]:


packing=PassiveAggressiveClassifier(max_iter=45)
packing.fit(tfidf_train,y_train)
y_pred=packing.predict(tfidf_test)
auc_score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(auc_score*100,2)}%')


# ### Here the accuracy score of the Passive Aggressive Classifier is 93.9% ~ 94% which is considered to be highly accurate so we can use this model on our dataset

# In[10]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# ### The Number of true positive for our model is 878, number of true negative is 907,number of false positive is 74 and false negative is 42

# ### 2.  Using Count Vectorizer and Multinomial NB

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
test = MultinomialNB()


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')


# In[18]:


count_train = count_vectorizer.fit_transform(x_train) 


# In[19]:


count_test = count_vectorizer.transform(x_test)


# In[20]:


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())


# In[21]:


test.fit(count_train, y_train)


# In[22]:


prediction = test.predict(count_test)


# In[28]:


import sklearn.metrics as metrics
auc_score_count = metrics.accuracy_score(y_test, prediction)
print(f'Accuracy: {round(auc_score_count*100,2)}%')


# ### Here the accuracy score of the Multinomial Naive Bayes Classifer is 88.85% which is considered less than than of Passive Aggressive Classifer.So, Passive Aggressive classifer proves to be more suitable for this dataset than the significant other
