#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#sum all of the columns
sd = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')

#הצגת העמודה השלישית של מאמר 1987_2
y = sd.iloc[:, [2]].values
#לולאה שמדפיסה את 10 המילים שמופיעות הכי הרבה במאמר את המיקומים שלהם וכמות הפעמים הופיעו
while (max(y)>15):
        max_num=int(max(y))
        index = list(y).index(max_num) 
        print("The index is:",index, "currant max number:",max_num, "word:",sd.iloc[index][0])
        y=np.delete(y,index)

       


# In[ ]:





# In[ ]:




