#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# DATA PREPROCESSING
# 

# In[23]:


#load the dataset to pandas dataframe
raw_mail_data = pd.read_csv("C:/Users/HP/OneDrive/Desktop/spam.csv",encoding=('ISO-8859-1'),low_memory =False)
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')


# In[24]:


mail_data.shape


# In[25]:


mail_data.head()


# In[26]:


#label spam mail as 0 ; non spam mail as 1.
mail_data.loc[mail_data['v1']== 'spam','v1',] = 0
mail_data.loc[mail_data['v1']== 'ham','v1',] = 1


# In[27]:


mail_data.head()


# In[28]:


X = mail_data['v2']
Y = mail_data['v1']


# In[29]:


print(X)
print('...........')
print(Y)


# In[30]:


# split the data as train data and test data
X_train, X_test,Y_train,Y_test = train_test_split(X ,Y ,train_size=0.8, test_size=0.2,random_state=3)


# In[31]:


# transform the text data to feature vectors that can be used as input to  the SVM model using TfidfVectorizer
# convert the text to lower case letters
feature_extraction = TfidfVectorizer(min_df=1 , stop_words='english',lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[32]:


#training the suppport vector machine model with training data
model = LinearSVC()
model.fit(X_train_features,Y_train)


# In[33]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[34]:


print('Accuracy_on_Training_data :' , accuracy_on_training_data)


# In[35]:


#prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)


# In[36]:


print("Accuracy on test data:" , accuracy_on_test_data)


# In[37]:


input_mail = ["Hello Harsha, We recently invited you to participate in this survey but it seems you haven't been able to begin it yet. There is still time and we still need a number of opinions so please take part today and we'll thank you with 2,000 Points."]
# convert text to feature vectors
input_mail_features = feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_mail_features)
print(prediction)

if(prediction[0]==1):
  print('HAM mail')
else:
  print('Spam mail')


# In[ ]:




