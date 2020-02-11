import gensim
from sense2vec import Sense2Vec
import random

w2v_model = "D:\WikiData\Trained\word2vec\\10GB\word2vec.model"
s2v_model = "D:\WikiData\Trained\sense2vec\RedditVecs\sense2vec-vectors"


# w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

f = open('ran_verbs.txt', 'r')
verbs = f.read().splitlines()
f.close()


for i in range(50):
    verb1 = random.choice(verbs)
    verb2 = random.choice(verbs)
    print(verb1, verb2, str(s2v.similarity([verb1 + '|VERB'], [verb2 + '|VERB'])))

