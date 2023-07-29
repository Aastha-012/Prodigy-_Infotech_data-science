#!/usr/bin/env python
# coding: utf-8

# # PRODIGY INFOTECH

# # Aastha Singla

# ## TASK : Create a bar chart or histogram to visualise the distribution of ages or gender in a population.

# In[13]:


pip install matplotlib


# In[2]:


import matplotlib


# In[14]:


import pandas as pd
age = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60",
       "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "91+"]
male = [3, 3.3, 3.4, 3.2, 3, 3.3, 3.7, 3.9, 3.5, 3.1, 3.4, 2.8, 2.4, 2.1, 1.8, 1.4, 0.8, 0.4, 0.1]
female = [2.9, 3.1, 3.2, 3, 3, 3.4, 3.9, 4, 3.6, 3.2, 3.5, 2.9, 2.5, 2.3, 2.2, 2, 1.4, 0.9, 0.5]

population_df = pd.DataFrame({"Age": age, "Male": male, "Female": female})

population_df["Female_Left"] = 0
population_df["Female_Width"] = population_df["Female"]

population_df["Male_Left"] = -population_df["Male"]
population_df["Male_Width"] = population_df["Male"]

population_df


# In[11]:


female_color = "#ee7a87"
male_color = "#4682b4"
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))

plt.barh(y=population_df["Age"], width=population_df["Female_Width"], color="#ee7a87", label="Female");
plt.barh(y=population_df["Age"], width=population_df["Male_Width"], left=population_df["Male_Left"]),
plt.text(-5, 17, "Male", fontsize=25, fontweight="bold");
plt.text(4, 17, "Female", fontsize=25, fontweight="bold");

for idx in range(len(population_df)):
    plt.text(x=population_df["Male_Left"][idx]-0.1, y=idx, s="{} %".format(population_df["Male"][idx]),
             ha="right", va="center",
             fontsize=15, color="#4682b4");
    plt.text(x=population_df["Female_Width"][idx]+0.1, y=idx, s="{} %".format(population_df["Female"][idx]),
             ha="left", va="center",
             fontsize=15, color="#ee7a87");

plt.xlim(-7,7);
plt.xticks(range(-7,8), ["{} %".format(i) for i in range(-7,8)]);

plt.legend(loc="best");

plt.xlabel("Percent (%)", fontsize=16, fontweight="bold")
plt.ylabel("Age Range", fontsize=16, fontweight="bold")
plt.title("US Population Pyramid Chart", loc="left", pad=20, fontsize=25, fontweight="bold");



# In[ ]:




