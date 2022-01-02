#!/usr/bin/env python
# coding: utf-8

# In[2]:


#I used the official userguides of mlxtend for apriori and association_rules for this project.


# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install apyori')
get_ipython().system('{sys.executable} -m pip install mlxtend')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


# In[5]:


marketData = pd.read_csv(r"C:\Users\nefle\mbo.csv", header=None)
marketData.head()


# In[9]:


itemsets = []
for i in range(0, 7501):
    itemsets.append([str(marketData.values[i,j]) for j in range(0, 20)])
for i,j in enumerate(itemsets):
    while 'nan' in itemsets[i]: itemsets[i].remove('nan')
#data are now in the form of a list of lists eg. list of itemsets
itemsets[1:10]


# In[15]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


# In[16]:


te = TransactionEncoder()
te_ary = te.fit(itemsets).transform(itemsets)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# In[17]:


#generating frequent itemsets
from mlxtend.frequent_patterns import apriori

frequent_itemsets0 = apriori(df, min_support=0.02, use_colnames = True)
frequent_itemsets0


# In[18]:


#generating rules
from mlxtend.frequent_patterns import association_rules

rules0 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.15)
rules0


# In[19]:


#gathering data for minsup vs. frequent datasets: i'll change minsup each time and gather the number of frequent datasets
#by getting the size of the dataframe

#1st data point
frequent_itemsets1 = apriori(df, min_support=0.01, use_colnames = True)
frequent_itemsets1.shape[0]


# In[20]:


#2nd data point
frequent_itemsets2 = apriori(df, min_support=0.05, use_colnames = True)
frequent_itemsets2.shape[0]


# In[21]:


#3rd data point
frequent_itemsets3 = apriori(df, min_support=0.1, use_colnames = True)
frequent_itemsets3.shape[0]


# In[22]:


#4th data point
frequent_itemsets4 = apriori(df, min_support=0.15, use_colnames = True)
frequent_itemsets4.shape[0]


# In[23]:


#5th data point
frequent_itemsets5 = apriori(df, min_support=0.2, use_colnames = True)
frequent_itemsets5.shape[0]


# In[24]:


import matplotlib.pyplot as plt
import numpy as np
x1 = np.array([0.01, 0.05, 0.1, 0.15, 0.2])
y1 = np.array([257, 28, 7, 5, 1])
plt.plot(x1, y1)
plt.xlabel("Minimum Support")
plt.ylabel("Number of Frequent Datasets")
plt.show()


# In[25]:


#gathering data for confidence vs. number of association rules: i'll change confidence each time and gather the number
#of frequent datasets by getting the size of the dataframe. I'm keeping the same chosen frequent itemset for each 
#calculation.

#1st data point
rules1 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.05)
rules1.shape[0]


# In[26]:


#2nd data point
rules2 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.10)
rules2.shape[0]


# In[27]:


#3rd data point
rules3 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.20)
rules3.shape[0]


# In[28]:


#4th data point
rules4 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.30)
rules4.shape[0]


# In[29]:


#5th data point
rules5 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.40)
rules5.shape[0]


# In[30]:


#6th data point
rules5 = association_rules(frequent_itemsets0, metric="confidence", min_threshold=0.50)
rules5.shape[0]


# In[50]:


import matplotlib.pyplot as plt
import numpy as np
x2 = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
y2 = np.array([100, 94, 55, 20, 3, 0])
plt.plot(x2, y2)
plt.xlabel("Threshold Confidence")
plt.ylabel("Number of Association Rules")
plt.show()

