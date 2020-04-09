import gensim
from gensim.test.utils import datapath
from gensim import utils


class MyCorpus(object):
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('E:\WikiData\TENGB\wiki10.txt')
        for line in open(corpus_path, errors="ignore"):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


data = MyCorpus()
model = gensim.models.Word2Vec(data, min_count=1, size=100, window=3, sg=1)

model.save("word2vec.model")

