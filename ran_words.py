import gensim
from sense2vec import Sense2Vec
import random
from nltk.corpus import wordnet as wn
import re

w2v_model = "D:\WikiData\Trained\word2vec\\10GB\word2vec.model"
s2v_model = "D:\WikiData\Trained\sense2vec\RedditVecs\sense2vec-vectors"


# w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

f = open('ran_verbs.txt', 'r')
verbs = f.read().splitlines()
f.close()


def wordnet_first_last(word):
    synsets = wn.synsets(word, pos=wn.VERB)
    first_synset = synsets[0]
    first_synset = re.search("'.*'", str(first_synset)).group(0)
    last_synset = synsets[-1]
    last_synset = re.search("'.*'", str(last_synset)).group(0)
    first_synset = [str(lemma.name()) for lemma in wn.synset(first_synset.replace("'", "")).lemmas()]
    last_synset = [str(lemma.name()) for lemma in wn.synset(last_synset.replace("'", "")).lemmas()]
    first_word_first_synset = first_synset[0]
    last_word_last_synset = last_synset[-1]
    return [first_word_first_synset, last_word_last_synset]


# for i in range(50):
#     verb1 = random.choice(verbs)
#     verb2 = random.choice(verbs)
#     print(verb1, verb2, str(s2v.similarity([verb1 + '|VERB'], [verb2 + '|VERB'])))

for i in range(10):
    verb = random.choice(verbs)
    wordnet_verbs = wordnet_first_last(verb)
    print(wordnet_verbs[0], wordnet_verbs[1], str(s2v.similarity([wordnet_verbs[0] + '|VERB'],
                                                                 [wordnet_verbs[1] + '|VERB'])))

