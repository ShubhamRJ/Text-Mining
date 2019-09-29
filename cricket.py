#CORPUS_PATH = "E:\\Users\\Debjani\\Documents\\Debjani's Thesis\\DataSets\\bbcsport\\cricket\\"
#CORPUS_PATH = "E:\\Users\\Debjani\\Documents\\Debjani's Thesis\\DataSets\\CompiledSportsData\\"
CORPUS_PATH = "E:\\Users\\Debjani\\Documents\\DebjanisThesis\\DataSets\\ModifiedSportsData\\"
import os
import time
import scipy as sy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as ny
from sklearn.cluster import KMeans
flnames = os.listdir(CORPUS_PATH)
all_texts = []
start = time.clock()
'''
for f in flnames:
    fp = open(CORPUS_PATH+f,"r")
    all_texts.append(fp.read())
'''
all_texts = [open(CORPUS_PATH+f).read() for f in flnames]    
end_time = time.clock()
print(all_texts[0])
print(end_time-start)
tfidf = TfidfVectorizer(all_texts, stop_words='english')
vectors = tfidf.fit_transform(all_texts)
all_words = tfidf.get_feature_names()
vectors_dense = vectors.todense()
svd = TruncatedSVD(n_components=2)
vectors_dense_svd = svd.fit_transform(vectors_dense)
vectors_dense_svd = Normalizer().fit_transform(vectors_dense_svd)
print(vectors_dense_svd.shape)
#sy.savetxt("E:\\Users\\Debjani\\Documents\\Debjani's Thesis\\DataSets\\vector_cricket_svd.txt",vectors_dense_svd,delimiter="\t")
#sy.savetxt("E:\\Users\\Debjani\\Documents\\Debjani's Thesis\\DataSets\\vector_combined_sports_svd.txt",vectors_dense_svd,delimiter="\t")
sy.savetxt("E:\\Users\\Debjani\\Documents\\DebjanisThesis\\DataSets\\vector_modified_sports_svd.txt",vectors_dense_svd,delimiter="\t")
start = time.clock()
dist = ny.array([[euclidean_distances(vectors_dense_svd[i], vectors_dense_svd[j])
 for j in ny.arange(len(all_texts))]
  for i in ny.arange(len(all_texts)) ]).reshape((265, 265))
end_time = time.clock()
print(end_time - start)  
print(dist)    