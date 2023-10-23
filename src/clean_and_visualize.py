#!/usr/bin/env python
# coding: utf-8

# # 1. Exploring the dataset

# In[1]:


# Import libraries

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np
import os
import re


# In[2]:


# Import file and create dataframe

csv_file = 'attacks.csv'
sharks = pd.read_csv(csv_file, encoding='ISO-8859-1')
sharks


# In[3]:


# Check the number of rows and columns in the dataframe

sharks.shape


# In[4]:


# Check the datatypes of the dataframe

sharks.dtypes


# In[5]:


# Check the names of the columns

sharks.columns


# # 2. Data pre-processing

# In[6]:


# Rename the columns with a comprehension list

sharks.columns = [i.lower().replace(" ", "_") for i in sharks.columns]
sharks


# In[7]:


#Rename the columns of sex_, species_ and fatal_(y/n)

sharks.rename(columns={"sex_":"sex"}, inplace=True)
sharks.rename(columns={"species_":"species"}, inplace=True)
sharks.rename(columns={"fatal_(y/n)":"fatal"}, inplace=True)
sharks


# In[8]:


# Check for null values

sharks.isna().sum()


# In[9]:


# Dropping the rows where all the columns have null values

sharks = sharks.dropna(how='all')


# In[10]:


# Checking if there are less missing values now

sharks.isna().sum()


# In[11]:


# Checking for duplicated data

print(f"Dataset has {sharks.duplicated().sum()} duplicated data.")


# In[12]:


# Dropping the duplicates

sharks = sharks.drop_duplicates() 

# Checking if the previous step has worked:

print(f"Dataset has {sharks.duplicated().sum()} duplicated data.")


# In[13]:


# Exploring the sex column

sharks.value_counts('sex')


# # 3. Transforming the data

# In[14]:


# Checking how are the values of the 'sex' column, so that I can modify them afterwards.

set(sharks.sex)


# In[15]:


# Creating a function so that the values of sex only return 'M', 'F' or 'unknown'

def clean_sex(x):
    if x != 'M' and x != 'F' and x != 'M ':
        return "unknown"
    if x == 'M ':
        return 'M'
    else:
        return x


# In[16]:


sharks['sex']=sharks['sex'].apply(clean_sex)


# In[17]:


# checking if the function has worked:

set(sharks.sex)


# In[18]:


pd.DataFrame(sharks.value_counts('sex')).sample(3)


# In[19]:


set(sharks.type)


# ## 3.1. Segmentation for Provoked Sharks Attacks since 2000 until nowadays

# In[20]:


# I want to analyze the people who provoked shark accidents, that's why I'm filtering the dataset where the type of attack is: provoked.

provoked_df = sharks[(sharks['type'] == 'Provoked')]
provoked_df


# In[21]:


set(provoked_df.year)


# In[22]:


def modern_years(x):
    x = int(x)
    if x >= 2000:
        return x
    else:
        return "old years"


# In[23]:


provoked_df['year'] = provoked_df['year'].apply(modern_years)


# In[24]:


# I'm going to create a subset considering only provoked shark attacks since the year 2000

provoked_df = provoked_df[provoked_df['year'] != 'old years']
provoked_df


# In[25]:


# Checking if the function has worked

set(provoked_df.year)


# In[26]:


# Now I'm going to create a subset with only the columns that are relevant for my research.

provoked_df = provoked_df[['sex', 'age', 'country', 'area', 'location', 'activity', 'fatal', 'time', 'year']]
provoked_df


# ### 3.1.1. How many males and females are among the people who provoked shark attacks?

# In[27]:


pd.DataFrame(provoked_df.value_counts('sex')).sample(3)


# In[28]:


sex_prov = provoked_df["sex"].value_counts()
print(sex_prov)

sex_prov.plot.pie(autopct="%.1f%%");


# ### 3.1.2. What is the mean age of the people who provoked the attacks?

# In[29]:


# Checking how are the values of the column 'age' so that I can modify them afterwards. 

set(provoked_df.age)


# In[30]:


# Creating a function to clean the values of the column 'age'. 

def clean_age(x):
    if pd.isna(x):
        return "unknown"
    if x in ["teen", "Teens", "Teens"]:
        return "15"
    if x == "middle-aged":
        return "50"
    if x in ["adult", "(adult)"]:
        return 35  
    if x == "18 months":
        return "2"
    
    age_av = re.findall(r'(\d{1,2})\s*(&|or|to)\s*(\d{1,2})', str(x))
    if age_av:
        average_ages = [round((int(match[0]) + int(match[2])) / 2) for match in age_av]
        return str(average_ages[0])
    
    age_match = re.search(r'\d{1,2}', str(x))
    if age_match:
        return age_match.group()
    
    return "unknown"


# In[31]:


provoked_df.loc[:, 'age'] = provoked_df['age'].apply(clean_age)


# In[32]:


set(provoked_df.age)


# In[33]:


age_prov = provoked_df["age"].value_counts()
age_prov


# In[34]:


# I create a counter to check hoy many unknown age values I have compared to the known. 

age_counter = provoked_df['age'].apply(lambda x: x != 'unknown').value_counts()
age_counter


# In[35]:


'''I need to compare the sex with the age of the people who provoked shark attacks. 
So I'm going to create a new data frame where I drop the rows if there is a useless value in the column of 'age'.
'''


# In[36]:


provoked_df2 = provoked_df[provoked_df['age'].notna()]


# In[37]:


provoked_df3 = provoked_df2[provoked_df2['age'] != 'unknown']


# In[38]:


set(provoked_df3.age)


# In[39]:


provoked_df3['age'] = provoked_df3['age'].astype(float)

sns.histplot(data=provoked_df3, x='age', color='red')


# In[40]:


# Now I can see the average age of men and women (and unknown sex) that provoked shark attacks. 

sns.histplot(data=provoked_df3, x='age', hue="sex", multiple="stack")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Men and women that provoked shark attacks')
plt.show()


# ### 3.1.3. Where are most of the people who provoked shark attacks from?

# In[41]:


# Checking how are the values of the 'country' column

set(provoked_df3.country)


# In[42]:


# Creating a function that standardizes Scotland and England with the United Kingdom. (I feel sorry for the Scottish people though)

def clean_uk(x):
    if x == 'SCOTLAND' or x == 'ENGLAND':
        return 'UNITED KINGDOM'
    else:
        return x


# In[43]:


# Checking if the function works

provoked_df3.loc[:, 'country'] = provoked_df3['country'].apply(clean_uk)


# In[44]:


country_prov = provoked_df3['country'].value_counts()
country_prov


# In[45]:


counts = provoked_df3['country'].groupby(provoked_df3['country']).transform('count')

# To filter the countries, I create a boolean mask to check in which countries there have been more than 1 provoked attack:

mask = counts > 1
provoked_df3['country'] = provoked_df3['country'][mask]


# In[46]:


country_prov = provoked_df3['country'].value_counts()
country_prov


# In[47]:


country_prov.plot(kind='bar', figsize=(13,7))
plt.title('Distribution of Provoked Shark Attacks')
plt.xlabel('Countries')
plt.ylabel('Number of Provoked Attacks')
plt.xticks(rotation=45) 
for index, value in enumerate(country_prov):
    plt.text(index, value + 4, str(value), ha='center')
plt.show()


# In[48]:


# Now I want to correlate the columns of sex and country

grouped = provoked_df3.groupby(['country', 'sex']).size().unstack()
grouped.plot(kind='bar', figsize=(10,6), stacked=True, colormap='tab20c')


# In[49]:


# I'm going to check which are the areas in the first two countries (where there have been more provoked shark attacks)
 
provoked_areas = provoked_df3[provoked_df3['country'].isin(['USA', 'AUSTRALIA'])]
provoked_areas


# In[50]:


areas = provoked_areas["area"].value_counts()
areas


# In[51]:


def clean_areas(x):
    if x == 'Westerm Australia':
        return 'Western Australia'
    else:
        return x


# In[52]:


provoked_df3.loc[:, 'area'] = provoked_df3['area'].apply(clean_areas)
provoked_areas.loc[:, 'area'] = provoked_areas['area'].apply(clean_areas)
areas = provoked_areas["area"].value_counts()
areas


# In[53]:


# Exploring the locations where there have been more provoked shark attacks, just for the fun :)

set(provoked_areas.location)


# In[54]:


# And now I'm checking which are the areas where there have been more provoked shark attacks. 

grouped2 = provoked_areas.groupby(['area', 'country']).size().unstack()
grouped2.plot(kind='bar', figsize=(10,6), stacked=True, colormap='inferno')


# ### 3.1.4. Which activities were doing most of the people who provoked shark attacks?

# In[55]:


# Checking how are the values of the column 'age' so that I can modify them afterwards. 
# Here I work with the previous dataframe (provoked_df instead of provoked_df3), otherwise I'd loose interesting information. 
# I'll see later what can I do in case I want to relate the activities with ages. 

set(provoked_df.activity)


# In[56]:


provoked_df = provoked_df[provoked_df['activity'].notna()]


# In[57]:


# Creating a function to clean the values of the 'activity' column

def func_(dict_, string):
    for key, words in dict_.items():
        if any(word in str(string) for word in words):
            return key


# In[58]:


def clean_activities(activity):
    activities = {
        'boat': ['boat', 'boating', 'racing', 'barqued', 'sinking', 'ship', 'wreck', 'dhow', 'kayak', 'canoa', 'raft', 'cutter', 'bark', 'submarine'],
        'fishing': ['fishing', 'fish', 'spearfishing', 'netting', 'wade-fishing', 'hunting', 'fishingat', 'shrimping', 'gigging', 'picking', 'hook', 'net'],
        'air_accidents': ['aircraft', 'airliner', 'constellation'],
        'swimming': ['swimming', 'riding'],
        'diving': ['diving', 'diver', 'photographing', 'dive', 'skindiving'],
        'surf': ['surfing', 'surf', 'boogie', 'surf-skiing', 'paddeling', 'boarding', 'board', 'overboard', 'treading'],
        'bathing': ['playing', 'bath', 'bathing', 'crouching', 'floating', 'standing', 'sitting', 'dangling']
    }

    for key, word_list in activities.items():
        if any(keyword in str(activity).lower() for keyword in word_list):
            return key
    return "shark interaction"


# In[59]:


provoked_df.loc[:, 'activity'] = provoked_df['activity'].apply(clean_activities)


# In[60]:


# Checking if the function has filtered the activities

set(provoked_df.activity)


# In[61]:


# I create a counter to check hoy many unknown activities I have compared to the specific ones (disaster again)

act_counter = provoked_df['activity'].apply(lambda x: x != "shark interaction").value_counts()
age_counter


# In[62]:


# Check which are the activities that lead to more provoked shark attacks. 

act_prov = provoked_df["activity"].value_counts()
print(act_prov)

act_prov.plot.pie(autopct="%.1f%%");


# In[63]:


# Now I want to correlate the columns of activity and sex

grouped_act = provoked_df.groupby(['activity', 'sex']).size().unstack()
grouped_act.plot(kind='bar', stacked=True, colormap='tab20c')


# In[82]:


# Now I want to correlate the columns of activity and age

provoked_df3.loc[:, 'activity'] = provoked_df3['activity'].apply(clean_activities)

plt.figure(figsize=(13, 6))

sns.histplot(data=provoked_df3, x='age', hue="activity", multiple="stack")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('The ages and activities of people that provoked shark attacks')
plt.show()


# ### 3.1.5. How many provoked shark attacks resulted lethal?

# In[64]:


# Checking how are the values of the column 'fatal' so that I can modify them afterwards. 

set(provoked_df.fatal)


# In[65]:


# creating a function so that the values of the column 'fatal' are "Y", "N" or "unknown"

def clean_fatal(x):
    if x != "N" and x != "Y" and x != "N " and x != " N" and x != "y" and x != "M":
        return "unknown"
    elif x == "N " or x == " N" or x =="M":
        return "N"
    elif x == "y":
        return "Y"
    else:
        return x


# In[66]:


provoked_df.loc[:, 'fatal'] = provoked_df['fatal'].apply(clean_fatal)


# In[67]:


set(provoked_df.fatal)

# It worked :) 


# In[68]:


fatal_prov = provoked_df["fatal"].value_counts()
print(fatal_prov)

fatal_prov.plot.pie(autopct="%.1f%%");

# Only 3.3% of the provoked sharks attacks resulted lethal, and 1.05 % are unknown 


# In[69]:


# I'm going to check which activity provoked more fatal attacks. 

grouped_fatal = provoked_df.groupby(['activity', 'fatal']).size().unstack()
grouped_fatal.plot(kind='bar', stacked=True, colormap='tab20c')


# In[70]:


plt.figure(figsize=(10, 6))

sns.histplot(data=provoked_df, x='activity', hue="fatal", multiple="stack")
plt.xlabel('Activity')
plt.ylabel('Frequency')
plt.title('Which activities are more likely to be fatal?')
plt.show()

