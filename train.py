import gensim
from gensim.test.utils import datapath
from gensim import utils
import warnings
import pickle
import numpy as np



#warnings.filterwarnings(action='ignore')

#stop_words = stopwords.words('english')

# data = []
#
# line_no = 0
#
# with open("D:\\WikiData\FIVEGB\wiki5.txt", "r", encoding='utf-8') as infile:
#     for line in infile:
#         line_no += 1
#         yield utils.simple_preprocess(line)
#         if(line_no % 10000 == 0):
#             print(line_no)


#with open("D:\\WikiData\FIVEGB\DataList.txt", "w", encoding='utf-8') as outfile:
#    pickle.dump(data, outfile)

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('D:\WikiData\TENGB\wiki10.txt')
        for line in open(corpus_path, errors="ignore"):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


data = MyCorpus()
model = gensim.models.Word2Vec(data, min_count=1, size=100, window=3, sg=1)

model.save("word2vec.model")

