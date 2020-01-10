#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#sum all of the columns
sd = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')
#sd=data.head(11464)
# 1987_2 הצגת מילה ונתוני העמודה הרלוונטית במקרה זה המילה וכמה פעמים מופיעה במאמר 
#x = sd.iloc[:, [0,2]].values
#print(x)
#הדפסה של אותו דבר בלי העמודה הראשונה עם המילים
#y = sd.iloc[:, [2]].values
#print(y)
# max_num=max(y)
# #- learning תמצא את המילה שמופיעה הכי הרבה פעמים
# print(max_num)
# #תמצא מיקום האינדקס ותוריד את המילה
# index = list(y).index(max_num)
# print(index)
# y=np.delete(y,index)
# #המילה השניה המקסימלית-noice
# max_num2=max(y)
# print(max_num2)
# #תמצא מיקום האינדקס ותוריד את המילה
# index2 = list(y).index(max_num2)
# print(index2)
# y=np.delete(y,index2)
# #synapse המילה הבאה המקסימלית
# max_num3=max(y)
#print(max_num3)

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




