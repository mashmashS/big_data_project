#!/usr/bin/env python
# coding: utf-8

# In[8]:


##### import numpy as np
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('C:\\Datasets\\2015_reversed.csv')
#st=store_data.tail(30)
#st=store_data.head(30)
st=store_data
print(st)

records = []
for i in range(1, 30):
    records.append([str(st.values[i,j]) for j in range(1000,1050)])#רשימה חדשה עם כל הפריטים שאני יכול למצוא. תהפוך לסטרינג של כל הערכים במיקומים אי ג'י(מטריצה) והעמודות בין 1 ל20
print(records)

association_rules = apriori(records, min_support=0.002, min_confidence=0.0045, min_lift=3, min_length=2)#תמצא קשר בעזרת אפריורי על הרקורדז שבניתי שהמינימים סופורט  הוא 0.00045 אחוז כלומר מחפש סלים שמקיימים שלפחות 0.00045% מהדאטה סט מכיל אותם, סלים הכל מגודל מסויים זה האורך
association_results = list(association_rules)
#print(len(association_rules))
print('/n')
print("length:",len(association_results[0]))
print(association_results[0])
print(association_results[1])
print(association_results[2])


# In[ ]:





# In[ ]:




