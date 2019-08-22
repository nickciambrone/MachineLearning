
# coding: utf-8

# In[4]:


import pandas as pd


# In[10]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')


# In[14]:


data.columns = ['Image.Var','Image.Skew','Image.Curt','Entropy','Class']
data.head()


# In[15]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


sns.countplot(x='Class',data=data)


# In[17]:


sns.pairplot(data,hue='Class')


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


scaler = StandardScaler()


# In[20]:


scaler.fit(data.drop('Class',axis=1))


# In[21]:


scaled_features = scaler.fit_transform(data.drop('Class',axis=1))


# In[22]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# In[23]:


X = df_feat


# In[24]:


y = data['Class']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[27]:


import tensorflow as tf


# In[28]:


df_feat.columns


# In[29]:


image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')


# In[30]:


feat_cols = [image_var,image_skew,image_curt,entropy]
feat_cols


# In[31]:


classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)


# In[32]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)


# In[33]:


classifier.train(input_fn=input_func,steps=500)


# In[34]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[35]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[36]:


note_predictions[0]


# In[37]:


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# In[38]:


from sklearn.metrics import classification_report,confusion_matrix


# In[39]:


print(confusion_matrix(y_test,final_preds))


# In[40]:


print(classification_report(y_test,final_preds))


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rfc = RandomForestClassifier(n_estimators=200)


# In[43]:


rfc.fit(X_train,y_train)


# In[44]:


rfc_preds = rfc.predict(X_test)


# In[45]:


print(classification_report(y_test,rfc_preds))


# In[46]:


print(confusion_matrix(y_test,rfc_preds))

