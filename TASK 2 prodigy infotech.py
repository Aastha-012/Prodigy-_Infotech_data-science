#!/usr/bin/env python
# coding: utf-8

# # PRODIGY INFOTECH

# ## Aastha Singla

# ## TASK : Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice as the TITANIC dataset from Kaggle.Explore the relationships between variables and identify patterns and trends in the data.

# In[1]:


pip install matplotlib


# In[18]:


pip install pandas


# In[17]:


pip install seaborn


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load the dataset into dataframe

# In[20]:


data = pd.read_csv(r'C:\Users\parveen\Desktop\train.csv')


# ## Data Cleaning

# In[21]:


print(data.isnull().sum())
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# ## Exploratory Data Analysis

# In[23]:


survival_rate = data['Survived'].value_counts(normalize=True) * 100
print(f"Survival Rate:\n{survival_rate}")


# ## Visualizing the survival rate 

# In[24]:


plt.bar(['Not Survived', 'Survived'], data['Survived'].value_counts())
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Survival Rate')
plt.show()


# In[30]:


pclass_survived = data.groupby('Pclass')['Survived'].mean() * 100
print(f"Survival Rate by Pclass:\n{pclass_survived}")


plt.bar(pclass_survived.index, pclass_survived.values)
plt.xlabel('Pclass')
plt.ylabel('Survival Rate (%)')
plt.title('Survival Rate by Pclass')
plt.show()


# In[31]:


sex_survived = data.groupby('Sex')['Survived'].mean() * 100
print(f"Survival Rate by Sex:\n{sex_survived}")



plt.bar(sex_survived.index, sex_survived.values)
plt.xlabel('Sex')
plt.ylabel('Survival Rate (%)')
plt.title('Survival Rate by Sex')
plt.show()


# ## Explorring the age distribution

# In[32]:


# Exploring the age distribution of passengers
plt.hist(data['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()


# ## Exploring the fare distribution of passengers

# In[33]:


plt.hist(data['Fare'], bins=20)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.title('Fare Distribution')
plt.show()


# In[ ]:




