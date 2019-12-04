import stanfordnlp
import gensim
from gensim.test.utils import datapath
from gensim import utils

#stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos') # This sets up a default neural pipeline in English
#doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
#print(*[word.text + "|" + word.upos for sent in doc.sentences for word in sent.words])


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('X:\Thesis\TinyWiki.txt')
        for line in open(corpus_path, errors="ignore"):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


data = MyCorpus()
for word in data:
    doc = nlp(word)
    output = open('X:\Thesis\parsed.txt')
    # doc.sentences[0].print_dependencies()
    output.write(*[word.text + "|" + word.upos])
    output.close()

#extract sent by sent from iter and pass to nlp()
#add parsed sen to file and continue