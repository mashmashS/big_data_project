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
print("length:",len(association_results))
print(association_results[0])
print(association_results[1])
print(association_results[2])

פלט:
    
   Unnamed: 0  abalone  abbeel  abbott  abbreviate  abbreviated  abc  abeles  \
0    2015_374  abalone     NaN     NaN         NaN          NaN  NaN     NaN   
1    2015_375      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
2    2015_376      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
3    2015_377      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
4    2015_378      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
5    2015_379      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
6    2015_380      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
7    2015_381      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
8    2015_382      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
9    2015_383      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
10   2015_384      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
11   2015_385      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
12   2015_386      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
13   2015_387      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
14   2015_388      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
15   2015_389      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
16   2015_390      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
17   2015_391      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
18   2015_392      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
19   2015_393      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
20   2015_394      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
21   2015_395      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
22   2015_396      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
23   2015_397      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
24   2015_398      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
25   2015_399      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
26   2015_400      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
27   2015_401      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
28   2015_402      NaN     NaN     NaN         NaN          NaN  NaN     NaN   
29   2015_403      NaN     NaN     NaN         NaN          NaN  NaN     NaN   

    abernethy  abilistic  ...  zhou  zhu  zien  zilberstein zones  zoo zoom  \
0         NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
1         NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
2         NaN        NaN  ...  zhou  NaN   NaN          NaN   NaN  NaN  NaN   
3         NaN        NaN  ...  zhou  NaN   NaN          NaN   NaN  NaN  NaN   
4         NaN        NaN  ...  zhou  NaN   NaN          NaN   NaN  NaN  NaN   
5         NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
6         NaN        NaN  ...   NaN  NaN  zien          NaN   NaN  NaN  NaN   
7         NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
8         NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
9         NaN        NaN  ...  zhou  NaN   NaN          NaN   NaN  NaN  NaN   
10        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
11        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
12        NaN        NaN  ...  zhou  NaN   NaN          NaN   NaN  NaN  NaN   
13        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
14        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
15        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
16        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
17        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
18        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
19        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
20  abernethy        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
21        NaN        NaN  ...  zhou  zhu   NaN          NaN   NaN  NaN  NaN   
22        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
23        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
24  abernethy        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
25        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
26        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
27        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
28  abernethy        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   
29        NaN        NaN  ...   NaN  NaN   NaN          NaN   NaN  NaN  NaN   

    zou zoubin  zurich  
0   NaN    NaN     NaN  
1   NaN    NaN     NaN  
2   NaN    NaN     NaN  
3   NaN    NaN     NaN  
4   NaN    NaN     NaN  
5   NaN    NaN     NaN  
6   NaN    NaN     NaN  
7   NaN    NaN     NaN  
8   NaN    NaN     NaN  
9   NaN    NaN     NaN  
10  NaN    NaN     NaN  
11  NaN    NaN     NaN  
12  NaN    NaN     NaN  
13  NaN    NaN     NaN  
14  NaN    NaN     NaN  
15  NaN    NaN     NaN  
16  NaN    NaN     NaN  
17  NaN    NaN     NaN  
18  NaN    NaN     NaN  
19  NaN    NaN     NaN  
20  NaN    NaN     NaN  
21  NaN    NaN     NaN  
22  NaN    NaN     NaN  
23  NaN    NaN     NaN  
24  NaN    NaN     NaN  
25  NaN    NaN     NaN  
26  NaN    NaN     NaN  
27  NaN    NaN     NaN  
28  NaN    NaN     NaN  
29  NaN    NaN     NaN  

[30 rows x 11464 columns]
[['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'binary', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'bility', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'big', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'binary', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'biostatistics'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'big', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'biometrika', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'binary', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'binomial', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'big', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'bing', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'biometrika', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'binary', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'big', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'big', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'binary', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'bins', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'big', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'biometrika', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'], ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan']]
/n
length: 3
RelationRecord(items=frozenset({'big', 'bins'}), support=0.034482758620689655, ordered_statistics=[OrderedStatistic(items_base=frozenset({'big'}), items_add=frozenset({'bins'}), confidence=0.16666666666666666, lift=4.833333333333333), OrderedStatistic(items_base=frozenset({'bins'}), items_add=frozenset({'big'}), confidence=1.0, lift=4.833333333333333)])
RelationRecord(items=frozenset({'binary', 'binomial'}), support=0.034482758620689655, ordered_statistics=[OrderedStatistic(items_base=frozenset({'binary'}), items_add=frozenset({'binomial'}), confidence=0.19999999999999998, lift=5.8), OrderedStatistic(items_base=frozenset({'binomial'}), items_add=frozenset({'binary'}), confidence=1.0, lift=5.8)])
RelationRecord(items=frozenset({'binary', 'bins'}), support=0.034482758620689655, ordered_statistics=[OrderedStatistic(items_base=frozenset({'binary'}), items_add=frozenset({'bins'}), confidence=0.19999999999999998, lift=5.8), OrderedStatistic(items_base=frozenset({'bins'}), items_add=frozenset({'binary'}), confidence=1.0, lift=5.8)])
​


