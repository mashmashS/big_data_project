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


output:
1987_1       787
1987_2      2167
1987_3      1171
1987_4      1582
1987_5      2275
            ... 
2015_399    2214
2015_400    1534
2015_401    2141
2015_402    2090
2015_403    2050
Length: 5811, dtype: int64
11040357
0        111
1        147
2        195
3         57
4         70
        ... 
11458     52
11459     75
11460    147
11461    220
11462    117
Length: 11463, dtype: int64
80080
51
