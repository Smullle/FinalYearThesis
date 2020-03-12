import stanfordnlp
from nltk.tag.stanford import StanfordNERTagger
import gensim
from gensim.test.utils import datapath
from gensim import utils

# jar = "D:\StanfordModels\stanford-english-corenlp-2018-10-05-models.jar"
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')  # This sets up a default neural pipeline in English


# ner_tagger = StanfordNERTagger(nlp, jar, encoding='utf8')
# doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
# print(*[word.text + "|" + word.upos for sent in doc.sentences for word in sent.words])


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('D:\WikiData\TENGB\wiki10.txt')
        for line in open(corpus_path, errors="ignore"):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


total_words = 0

data = MyCorpus()
output = open('D:\WikiData\Trained\sense2vec\Parsed\Parsed10.txt', "a")
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

# for block in data:
#     segments = ([block[x:x + 10] for x in range(0, len(block), 10)])
#     for segment in segments:
#         total_words += 10
#         doc = nlp(' '.join(segment))
#         output = open('D:\WikiData\Trained\sense2vec\Stanford\Parsed10.txt', "a")
#         # doc.sentences[0].print_dependencies()
#         for sent in doc.sentences:
#             for word in sent.words:
#                 output.write(*[word.text + "|" + word.upos + " "])
#
#         output.close()
#
#         if total_words % 100000 == 0:
#             print(total_words)
