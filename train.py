import gensim
import warnings
import numpy as np



warnings.filterwarnings(action='ignore')

data = open("D:\\wiki.txt", "r", encoding='utf-8')

model = gensim.models.Word2Vec(data, min_count=1, size=100, window=3)

model.save("word2vec.model")
