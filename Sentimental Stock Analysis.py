#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 


# In[14]:


df=pd.read_csv(r"C:\Users\Priya\Desktop\Projects-master\Projects\Stock-Sentiment-Analysis-master\Data.csv",encoding='ISO-8859-1')


# In[15]:


df.head()


# In[19]:


train=df[df['Date']<'20150101']
test=df[df['Date']>'20150101']


# In[20]:


#Removing Punctuation
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)


# In[21]:


#Renaming Columns to numbers
list1=[str(i) for i in range(25)]
data.columns=list1


# In[22]:


data.head()


# In[24]:


#Converting headlines to same lowercase letters
for i in list1:
    data[i]=data[i].str.lower()
data.head(2)


# In[26]:


' '.join(str(x) for x in data.iloc[0,0:25])


# In[28]:


headlines=[]
for i in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[i,0:27]))


# In[31]:


headlines[0]


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[33]:


countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)


# In[34]:


traindataset[0]


# In[37]:


#implementing randomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[54]:


test.head()


# In[41]:


#Converting test data
test_transform=[]

for i in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[i,0:26]))
test_dataset=countvector.transform(test_transform)


# In[42]:


#prediction for the test data
prediction=randomclassifier.predict(test_dataset)


# In[53]:


test.loc[3975,:]


# In[46]:


prediction


# In[ ]:





# In[55]:


# Classification report,Confusion Matrix ,accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[56]:


matrix=confusion_matrix(test['Label'],prediction)
print(matrix)
score=accuracy_score(test['Label'],prediction)
print(score)
report=classification_report(test['Label'],prediction)
print(report)


# In[57]:


## Convert using Tfidfvectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[58]:


tfidfvector=TfidfVectorizer(ngram_range=(2,2))
traindataset=tfidfvector.fit_transform(headlines)


# In[59]:


#implementing randomForest Classifier again
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[61]:


test_dataset=tfidfvector.transform(test_transform)
prediction=randomclassifier.predict(test_dataset)


# In[62]:


prediction


# In[63]:


matrix=confusion_matrix(test['Label'],prediction)
print(matrix)
score=accuracy_score(test['Label'],prediction)
print(score)
report=classification_report(test['Label'],prediction)
print(report)


# In[64]:


from sklearn.naive_bayes import MultinomialNB


# In[65]:


traindataset


# In[66]:


multi_naive=MultinomialNB()
multi_naive.fit(traindataset,train['Label'])


# In[67]:


multi_naive


# In[68]:


#Converting test data
test_transform=[]

for i in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[i,0:26]))
test_dataset=tfidfvector.transform(test_transform)
prediction=multi_naive.predict(test_dataset)


# In[69]:


prediction


# In[70]:


matrix=confusion_matrix(test['Label'],prediction)
print(matrix)
score=accuracy_score(test['Label'],prediction)
print(score)
report=classification_report(test['Label'],prediction)
print(report)


# In[ ]:




