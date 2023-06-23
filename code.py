#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Read the provided CSV file ‘data.csv’.

import pandas as pd
df = pd.read_csv('data.csv')     #reading csv file
df


# In[2]:


#Show the basic statistical description about the data.

df.describe()   #describe() results statistical description of data in data frame


# In[3]:


#Check if the data has null values.

df.isnull().any()  #check any column has null values


# In[4]:


#Replace the null values with the mean

mean=df['Calories'].mean()
df['Calories'].fillna(value=mean, inplace=True)  #replacing Nan values with particular columns mean value


# In[5]:


df.isnull().any()


# In[6]:


#Select at least two columns and aggregate the data using: min, max, count, mean.

df.agg({'Pulse' : ['min', 'max', 'count', 'mean'], 'Maxpulse' : ['min', 'max', 'count', 'mean'], 
        'Calories' : ['min', 'max', 'count', 'mean'] })
#agg method to aggreate operation on the dataframe


# In[7]:


#Filter the dataframe to select the rows with calories values between 500 and 1000. 

df[(df['Calories'] > 500) & (df['Calories'] < 1000)]   #'&' operator to filter the dataframe


# In[8]:


#Filter the dataframe to select the rows with calories values > 500 and pulse < 100.

df[(df['Calories'] > 500) & (df['Pulse'] < 100)]   # '&' operator is used to filter the data 


# In[9]:


#Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”.

df_modified = df[['Duration', 'Pulse', 'Calories']].copy()  #copy method to create an another data frome with specified columns from the original dataframe.
df_modified


# In[10]:


# Delete the “Maxpulse” column from the main df dataframe

df.pop('Maxpulse')   #pop method to remove a column from the data frame
df


# In[11]:


df.dtypes


# In[12]:


#Convert the datatype of Calories column to int datatype.

df['Calories'] = df['Calories'].astype(int)  #astype function converts one data type into another
df.dtypes


# In[13]:


#Using pandas create a scatter plot for the two columns (Duration and Calories).
df.plot.scatter(x='Duration', y='Calories')


# In[14]:


#(Glass Dataset)


# In[15]:


# 1. Implement Naïve Bayes method using scikit-learn library.
# a. Use the glass dataset available in Link also provided in your assignment.


# In[16]:


import numpy as np
import pandas as pd


# In[17]:


glass = pd.read_csv("glass.csv")
glass


# In[18]:


glass.info()


# In[19]:


# b. Use train_test_split to create training and testing part.


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_true = train_test_split(glass[::-1], glass['Type'], test_size = 0.2, random_state = 0)


# In[21]:


#2. Evaluate the model on testing part using score and classification_report(y_true, y_pred)


# In[22]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_true))


# In[23]:


# 1. Implement linear SVM method using scikit library
# a. Use the glass dataset available in Link also provided in your assignment.


# In[24]:


import numpy as np
import pandas as pd
glass = pd.read_csv("glass.csv")
glass


# In[25]:


# b. Use train_test_split to create training and testing part.


# In[26]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Support Vector Machine's 
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_true))


# In[27]:


# Do at least two visualizations to describe or show correlations in the Glass Dataset.


# In[28]:


import seaborn as sns   #For Visualisation import seaborn library
import matplotlib.pyplot as plt
sns.barplot(x = glass['Type'], y = glass['Ca']) 


# In[29]:


sns.catplot(data=glass, x="Type", y="K")


# In[30]:


sns.regplot(x="Type", y="Fe", data=glass);


# In[ ]:



