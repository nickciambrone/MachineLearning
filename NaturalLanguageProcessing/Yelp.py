
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[7]:


yelp.head()


# In[8]:


yelp.info()


# In[9]:


yelp.describe()


# In[10]:


yelp['text length'] = yelp['text'].apply(len)


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')


# In[13]:


sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')


# In[14]:


sns.countplot(x='stars',data=yelp,palette='rainbow')


# In[15]:


stars = yelp.groupby('stars').mean()
stars


# In[16]:


stars.corr()


# In[17]:


sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# In[18]:


yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[19]:


X = yelp_class['text']
y = yelp_class['stars']


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[21]:


X = cv.fit_transform(X)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[24]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[25]:


nb.fit(X_train,y_train)


# In[26]:


predictions = nb.predict(X_test)


# In[27]:


from sklearn.metrics import confusion_matrix,classification_report


# In[28]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[29]:


from sklearn.feature_extraction.text import  TfidfTransformer


# In[30]:


from sklearn.pipeline import Pipeline


# In[31]:


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[32]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[33]:


# May take some time
pipeline.fit(X_train,y_train)


# In[34]:


predictions = pipeline.predict(X_test)


# In[35]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

