#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("D:\\Datasets\\NIPS_1987-2015.csv", index_col ="word")

# retrieving row by loc method
first = data["1987_1"].head(300)
second = data["1987_2"].head(300)
Third = data["1987_3"].head(300)
Fourth=data["1987_4"].head(300)

#print(first, "\n\n\n", second)

#grouping all the articles to docs
docs = [pd.Series(first), pd.Series(second), pd.Series(Third), pd.Series(Fourth)]
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

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame
data = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
data.sort_values(by=["tfidf"],ascending=False)

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
row3=corr_matrix[2]
row4=corr_matrix[3]

#calculate distance beetween documents with cosine 
result1=1-spatial.distance.cosine(row1,row2)
result2=1-spatial.distance.cosine(row1,row3)
result3=1-spatial.distance.cosine(row1,row4)
result4=1-spatial.distance.cosine(row2,row3)
result5=1-spatial.distance.cosine(row2,row4)
result6=1-spatial.distance.cosine(row3,row4)

#dim = len(rep_docs)
#similarity = corr_matrix[dim:, :dim].mean()

#similarity to each one of the document
print(corr_matrix.mean(axis=1))
print("\n")
#similarity between all the documents
print(corr_matrix.mean())
print("\n")

#differnces between each 2 documents
print("similarity between 1987_1 and 1987_2",result1)
print("similarity between 1987_1 and 1987_3",result2)
print("similarity between 1987_1 and 1987_4",result3)
print("similarity between 1987_2 and 1987_3",result4)
print("similarity between 1987_2 and 1987_4",result5)
print("similarity between 1987_3 and 1987_4",result6)


output:

TF IDF values 
  (0, 50)	0.2362473525749386
  (0, 49)	0.2362473525749386
  (0, 32)	0.29964981070461477
  (0, 17)	0.2362473525749386
  (0, 12)	0.29964981070461477
  (0, 10)	0.29964981070461477
  (0, 9)	0.29964981070461477
  (0, 8)	0.29964981070461477
  (0, 4)	0.15636970200839706
  (0, 1)	0.5992996214092295
  (1, 52)	0.18638851106609494
  (1, 51)	0.25149593900107897
  (1, 47)	0.15760679488321205
  (1, 46)	0.07880339744160603
  (1, 45)	0.07880339744160603
  (1, 44)	0.07880339744160603
  (1, 43)	0.23641019232481808
  (1, 42)	0.07880339744160603
  (1, 41)	0.12425900737739662
  (1, 40)	0.15760679488321205
  (1, 39)	0.15760679488321205
  (1, 36)	0.23641019232481808
  (1, 33)	0.24851801475479324
  (1, 31)	0.05029918780021579
  (1, 30)	0.10059837560043158
  :	:
  (2, 14)	0.19044863286439437
  (2, 7)	0.19044863286439437
  (2, 4)	0.09938399726962793
  (2, 0)	0.3003037793344863
  (3, 53)	0.13432917629666874
  (3, 51)	0.08574057318532706
  (3, 48)	0.13432917629666874
  (3, 38)	0.13432917629666874
  (3, 35)	0.13432917629666874
  (3, 34)	0.13432917629666874
  (3, 33)	0.10590666551411057
  (3, 31)	0.08574057318532706
  (3, 30)	0.17148114637065412
  (3, 28)	0.3177199965423317
  (3, 27)	0.6716458814833437
  (3, 26)	0.13432917629666874
  (3, 24)	0.21181333102822114
  (3, 23)	0.2686583525933375
  (3, 21)	0.13432917629666874
  (3, 20)	0.13432917629666874
  (3, 19)	0.13432917629666874
  (3, 11)	0.2686583525933375
  (3, 6)	0.13432917629666874
  (3, 5)	0.13432917629666874
  (3, 4)	0.07009853675245468


{'ability': 1, 'abstract': 4, 'access': 8, 'accommodate': 9, 'according': 10, 'accumulated': 12, 'acknowledgement': 17, 'addison': 32, 'afosr': 49, 'air': 50, 'abilities': 0, 'absence': 2, 'absolute': 3, 'achieve': 13, 'ackley': 15, 'acknowledgment': 18, 'across': 22, 'activation': 24, 'activations': 25, 'adaptive': 28, 'added': 30, 'adding': 31, 'addition': 33, 'address': 36, 'adds': 39, 'adjacent': 40, 'adjust': 41, 'adjusting': 42, 'adjustment': 43, 'adjustments': 44, 'adopted': 45, 'advance': 46, 'advanced': 47, 'algorithm': 51, 'algorithms': 52, 'acceptable': 7, 'achieved': 14, 'acknowledge': 16, 'add': 29, 'addressed': 37, 'academic': 5, 'academy': 6, 'accordingly': 11, 'acoustic': 19, 'acoustical': 20, 'acoustics': 21, 'activa': 23, 'activity': 26, 'adaptation': 27, 'additions': 34, 'additive': 35, 'addresses': 38, 'affected': 48, 'aliasing': 53}


[[1.         0.00643037 0.12195961 0.01096129]
 [0.00643037 1.         0.28339349 0.28946682]
 [0.12195961 0.28339349 1.         0.12161638]
 [0.01096129 0.28946682 0.12161638 1.        ]]


[0.28483782 0.39482267 0.38174237 0.35551112]


0.3542284954893248


similarity between 1987_1 and 1987_2 0.04654511822859242
similarity between 1987_1 and 1987_3 0.23277190715573037
similarity between 1987_1 and 1987_4 0.0365670086486547
similarity between 1987_2 and 1987_3 0.5302675216394972
similarity between 1987_2 and 1987_4 0.5424373853966191
similarity between 1987_3 and 1987_4 0.29574870049655466


