#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("D:\\Datasets\\NIPS_1987-2015.csv", index_col ="word")

# retrieving row by loc method
first = data["1987_1"].head(300)
#second = data["1987_2"].head(300)
Third = data["1987_3"].head(300)
#Fourth=data["1987_4"].head(300)

#print(first, "\n\n\n", second)

#grouping all the articles to docs
#docs = [pd.Series(first), pd.Series(second), pd.Series(Third), pd.Series(Fourth)]
docs = [pd.Series(first), pd.Series(Third)]
rep_docs = [" ".join(i.repeat(i).index.values) for i in docs]
 
#instantiate CountVectorizer()
cv=CountVectorizer()
 
# this steps generates word counts for the words in rep_docs
word_count_vector=cv.fit_transform(rep_docs)

#how much docs in rows and columns
word_count_vector.shape
 
#compute Idf
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
 
# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
df_idf.sort_values(by=['idf_weights'])

#compute the tf-idf scores for any document or set of documents
#count matrix
count_vector=cv.transform(rep_docs) 
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)
#print(tf_idf_vector) 
    
#settings for count vectorizer
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(rep_docs)
 
#get the tf-idf scores of a set of documents.
fitted_vectorizer=tfidf_vectorizer.fit(rep_docs)
tfidf_vectorizer_vectors=fitted_vectorizer.transform(rep_docs) 

print("TF IDF values ")
print(tfidf_vectorizer_vectors)
print("\n")
print(tfidf_vectorizer.vocabulary_)
print("\n")

#correlation matrix between the documents 
vecs = tfidf_vectorizer.fit_transform(rep_docs)
corr_matrix = ((vecs * vecs.T).A)
print(corr_matrix)
print("\n")

#defining each row as document
row1=corr_matrix[0]
row2=corr_matrix[1]
#row3=corr_matrix[2]
#row4=corr_matrix[3]

#calculate distance beetween documents with cosine 
result1=1-spatial.distance.cosine(row1,row2)
#result2=1-spatial.distance.cosine(row1,row3)
#result3=1-spatial.distance.cosine(row1,row4)
#result4=1-spatial.distance.cosine(row2,row3)
#result5=1-spatial.distance.cosine(row2,row4)
#result6=1-spatial.distance.cosine(row3,row4)

dim = len(rep_docs)
similarity = corr_matrix[dim:, :dim].mean()


#differnces between each 2 documents
print("similarity between 1987_1 and 1987_2",result1)
#print("similarity between 1987_1 and 1987_3",result2)
#print("similarity between 1987_1 and 1987_4",result3)
#print("similarity between 1987_2 and 1987_3",result4)
#print("similarity between 1987_2 and 1987_4",result5)
#print("similarity between 1987_3 and 1987_4",result6)


# In[2]:


similarity


# In[ ]:




