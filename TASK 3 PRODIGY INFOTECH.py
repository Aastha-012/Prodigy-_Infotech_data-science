#!/usr/bin/env python
# coding: utf-8

# # PRODIGY INFOTECH

# ## AASTHA SINGLA 

# ## TASK : Build a decision tree classifier to predict whether a customer will purchase a product oor service based on the demographic and behavioral data . Use a dataset such as the Bank Marketing   UCI Machine Learning Repository.

# In[2]:


pip install sklearn


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = pd.read_csv(r'C:\Users\parveen\Desktop\bank.csv')
df.head()


# In[15]:


df.shape


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


df.isnull().sum()


# ## Visualize the Data

# In[96]:


plt.figure(figsize = (16,9))
sns.countplot(x = "job",data = df)


# In[20]:


sns.countplot(x = "job",data = df)


# In[21]:


sns.countplot(x = "marital",data = df)


# In[22]:


sns.countplot(x = "education",data = df)


# In[23]:


sns.countplot(x = "deposit",data = df)


# In[24]:


sns.countplot(x = "default",data = df)


# In[25]:


fig,axes = plt.subplots(nrows = 2,ncols = 2,figsize = (12,10))
df.plot(kind = "hist",y = "age",bins = 70,color = "blue",ax = axes[0][0])
df.plot(kind = "hist",y = "balance",bins = 10,color = "red",ax = axes[0][1])
df.plot(kind = "hist",y = "duration",bins = 60,color = "green",ax = axes[1][0])
df.plot(kind = "hist",y = "campaign",bins = 10,color = "orange",ax = axes[1][1])
plt.show()


# In[35]:


plt.figure(figsize = (16,9))
sns.pairplot(data = df,hue = "default")


# In[74]:


my_df=df.select_dtypes(exclude=[object])


# In[76]:


my_df.corr()


# In[77]:


plt.figure(figsize = (16,9))
sns.heatmap(my_df.corr(),annot = True)


# In[79]:


pip install scikit-learn


# ## Convert data from categorical to numerical

# In[80]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[81]:


df["job"] = le.fit_transform(df["job"])
df["marital"] = le.fit_transform(df["marital"])
df["education"] = le.fit_transform(df["education"])
df["deposit"] = le.fit_transform(df["deposit"])
df["default"] = le.fit_transform(df["default"])
df["loan"] = le.fit_transform(df["loan"])
df["contact"] = le.fit_transform(df["contact"])
df["poutcome"] = le.fit_transform(df["poutcome"])
df["housing"] = le.fit_transform(df["housing"])
df["month"] = le.fit_transform(df["month"])


# In[82]:


df.head()


# In[83]:


df.drop(["pdays","previous","poutcome"],axis = 1)
df.head()


# ## Split the data into train and test

# In[84]:


x = df.drop(["default"],axis = 1)
y = df["default"]


# In[85]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[86]:


x_train.shape


# In[87]:


x_test.shape


# In[88]:


y_train.shape


# In[89]:


y_test.shape


# ## Decision tree classifier model

# In[90]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()


# In[91]:


dc.fit(x_train,y_train)


# ## Predict the Model

# In[92]:


y_pred = dc.predict(x_test)
y_pred


# ## Evaluate the Model

# In[93]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[94]:


acc = accuracy_score(y_pred,y_test)*100
print("Accuracy Score :",acc)


# In[95]:


cm = confusion_matrix(y_pred,y_test)
print("Confusion Matrix:\n",cm)


# In[ ]:




