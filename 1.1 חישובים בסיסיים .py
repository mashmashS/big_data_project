#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#sum all of the columns
store_data = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')
#sum every column
sumCol=store_data.sum()
print(sumCol)
sumAll=np.sum(sumCol)
print(sumAll)
#a=np.squeeze(np.asarray(store_data))
#find the rows
x=np.sum(store_data,axis=1)
#print the rows
print(x)
#find the max
print(max(x))
print(min(x))



