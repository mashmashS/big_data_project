#!/usr/bin/env python
# coding: utf-8

# In[150]:



# %matplotlib inline
import numpy as np
import pandas as pd
import sklearn as sk
import math 

dt = pd.read_csv('D:\\Datasets\\dt.csv',index_col="word")
#dt.head(10)
#dt=data.head(18)
# retrieving row by loc method
#first=dt.iloc[[10,135,330,388], [1]].values
#print(first)
#second=dt.iloc[[10,135,330,388], [2]].values
#print(second)
first=dt["1987_1"]
second=dt["1987_2"]

row1=pd.DataFrame(first) 
row2=pd.DataFrame(second) 

#פונקציה שמחשבת TF
def computeTF(article):
    tf = []
    # Calculate the number of times a word appears in a document divded by the total number of words in the document
    for i in article: 
        cal=i/11463
        #i+=1
        #print(cal)
        tf.append(cal)
    return tf

#running our sentences through the tf function:
#header = data.iloc[0]
#Tf עבור המסמך הראשון
tfFirst = computeTF(first)
#Tf עבור המסמך השני
tfSecond = computeTF(second)
#Converting to dataframe for visualization
#dt.columns = dt.iloc[0]
#dt.rename(columns=dt.iloc[0])
Tf= pd.DataFrame([tfFirst, tfSecond])
#tf_df.rename(columns=dt.iloc[1])
print(Tf)
#head=dt["word"]
#tf_df = tf_df[1:]
#t = dt.iloc[0]
#tf_df.rename(columns = header)
#df=tf_df.iloc[0].values
#print(df)


#פונקציה שמחשבת IDF
def computeIdF(article1,article2):
    Idf = []
    sum=0
    for i,j in zip(article1,article2):
        # סיכום כמות הפעמים שהמילה מופיעה בשני המאמרים
        #sum=i+j
        #log of the number of documents divided by the number of documents that contain the word 
        #במקרה שהמילה מופיעה באחד המאמרים
        if (i>0 and j==0) or (i==0 and j>0):
            cal=math.log(2/1)
        #במקרה שהמילה מופיעה ב2 המאמרים התוצאה תמיד תתן 0
        if i>0 and j>0:
            cal=0
        #הכנסה של 1 היות וחלוקה ב0 לא אפשרית כשהמילה לא מופיע ב2 המאמרים 
        if i==0 and j==0:
            cal=1
        Idf.append(cal)
    return Idf

Idf=computeIdF(first,second)
IdfFrame= pd.DataFrame([Idf])
#print(Idf)
#print(IdfFrame)

#פונקציה שמחשבת TF-IDF
def computeTFIDF(tf, idf):
    TFIDF=Tf*Idf
    return TFIDF

TfidfA = computeTFIDF(tfFirst, Idf)
print(TfidfA)
TfidfB = computeTFIDF(tfSecond, Idf)
print(TfidfB)
TFIDFFrame = pd.DataFrame([TfidfA, TfidfB])   
print(TFIDFFrame)

output:

   0      1      2      3      4      5      6      7      8         9      \
0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  0.000000   
1    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  0.000087   

   ...  11453  11454  11455  11456  11457  11458  11459  11460  11461  11462  
0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  
1  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  

[2 rows x 11463 columns]
   0      1      2      3      4      5      6      7      8        9      \
0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  0.00000   
1    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  0.00006   

   ...  11453  11454  11455  11456  11457  11458  11459  11460  11461  11462  
0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  
1  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  

[2 rows x 11463 columns]
   0      1      2      3      4      5      6      7      8        9      \
0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  0.00000   
1    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  0.00006   

   ...  11453  11454  11455  11456  11457  11458  11459  11460  11461  11462  
0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  
1  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  

[2 rows x 11463 columns]
                                                   0
0     0      1      2      3      4      5      6...
1     0      1      2      3      4      5      6...
TFIDFFrame
0
0	0 1 2 3 4 5 6...
1	0 1 2 3 4 5 6...
Tf
0	1	2	3	4	5	6	7	8	9	...	11453	11454	11455	11456	11457	11458	11459	11460	11461	11462
0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.000000	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.000087	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2 rows × 11463 columns

Tf
0	1	2	3	4	5	6	7	8	9	...	11453	11454	11455	11456	11457	11458	11459	11460	11461	11462
0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.000000	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.000087	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2 rows × 11463 columns

IdfFrame
0	1	2	3	4	5	6	7	8	9	...	11453	11454	11455	11456	11457	11458	11459	11460	11461	11462
0	1	1	1	1	1	1	1	1	1	0.693147	...	1	1	1	1	1	1	1	1	1	1
1 rows × 11463 columns


