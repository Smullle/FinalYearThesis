import gensim
from sense2vec import Sense2Vec
import csv
from nltk.corpus import wordnet as wn


def sense_vec_query(verb):
    query = verb
    assert query in s2v
    # vector = s2v[query]
    # freq = s2v.get_freq(query)
    return s2v.most_similar(query, n=3)

# TODO: implement wordnet query function

pairs = open("X:\Thesis\Results\\pairs.txt", "w")
csv_data = csv.writer(pairs, delimiter=',')
w2v_model = "X:\Thesis\word2vec.model"
s2v_model = "X:\Thesis\sense2vec-vectors"

w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

csv_data.writerow(["Word Pairs", "word2vec", "sense2vec", "WordNet"])
csv_data.writerow(["walk-talk",
                   str(w2v.similarity("walk", "talk")),
                   str(s2v.similarity('walk|VERB', 'talk|VERB')),
                   wn.path_similarity("walk", "talk")])
csv_data.writerow(["run-walk",
                   str(w2v.similarity("run", "walk")),
                   str(s2v.similarity('run|VERB', 'walk|VERB')),
                   wn.path_similarity("run", "walk")])
csv_data.writerow(["walk-chair",
                   str(w2v.similarity("walk", "chair")),
                   str(s2v.similarity('walk|VERB', 'chair|NOUN')),
                   wn.path_similarity("walk", "chair")])
csv_data.writerow(["run-pen",
                   str(w2v.similarity("run", "pen")),
                   str(s2v.similarity('run|VERB', 'pen|NOUN')),
                   wn.path_similarity("run", "pen")])
csv_data.writerow(["",
                   str(w2v.similarity()),
                   str(s2v.similarity()),
                   wn.path_similarity()])
