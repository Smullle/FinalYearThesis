import stanfordnlp
from nltk.tag.stanford import StanfordNERTagger
import gensim
from gensim.test.utils import datapath
from gensim import utils


nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')  # This sets up a default neural pipeline in English


class MyCorpus(object):
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('E:\WikiData\TENGB\wiki10.txt')
        for line in open(corpus_path, errors="ignore"):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


total_words = 0

data = MyCorpus()
output = open('E:\WikiData\Trained\sense2vec\Parsed\Parsed10.txt', "a")
for line in data:
    total_words += len(line)
    try:
        if len(line) < 3000:
            doc = nlp(' '.join(line))
            for sent in doc.sentences:
                for word in sent.words:
                    output.write(*[word.text + "|" + word.upos + " "])
        else:
            segments = ([line[x:x + 3000] for x in range(0, len(line), 3000)])
            for segment in segments:
                doc = nlp(' '.join(segment))
                # doc.sentences[0].print_dependencies()
                for sent in doc.sentences:
                    for word in sent.words:
                        output.write(*[word.text + "|" + word.upos + " "])
    except AssertionError:
        print(len(line))

    print(total_words)

