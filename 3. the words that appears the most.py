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

       

output:
The index is: 5593 currant max number: 53 word: learning
The index is: 6771 currant max number: 37 word: nodes
The index is: 10255 currant max number: 27 word: symptoms
The index is: 6699 currant max number: 22 word: netw
The index is: 3722 currant max number: 20 word: fig
The index is: 8462 currant max number: 20 word: reid
The index is: 6718 currant max number: 18 word: neuromodulation
The index is: 11292 currant max number: 18 word: wedge
The index is: 7903 currant max number: 17 word: problem
The index is: 388 currant max number: 16 word: analog



