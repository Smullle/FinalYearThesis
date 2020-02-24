import gensim
from sense2vec import Sense2Vec
import random
from nltk.corpus import wordnet as wn
import re
import csv

w2v_model = "D:\\WikiData\\Trained\\word2vec\\10GB\\word2vec.model"
s2v_model = "D:\\WikiData\\Trained\\sense2vec\\RedditVecs\\sense2vec-vectors"

w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

f = open('D:\\wordlists\\ran_verbs.txt', 'r')
verbs = f.read().splitlines()
f.close()
f = open('D:\\wordlists\\nounlist.txt', 'r')
nouns = f.read().splitlines()
f.close()


def wordnet_first_last(word, pos):
    try:
        synsets = wn.synsets(word, pos=pos)
        first_synset = synsets[0]
        first_synset = re.search("'.*'", str(first_synset)).group(0)
        last_synset = synsets[-1]
        last_synset = re.search("'.*'", str(last_synset)).group(0)
        first_synset = first_synset[1:-1]
        last_synset = last_synset[1:-1]
        first_synset = [str(lemma.name()) for lemma in wn.synset(first_synset).lemmas()]
        last_synset = [str(lemma.name()) for lemma in wn.synset(last_synset).lemmas()]
        # remove quotes, but not apostrophes
        first_word_first_synset = first_synset[0]
        last_word_first_synset = first_synset[-1]
        last_word_last_synset = last_synset[-1]
        return [first_word_first_synset, last_word_first_synset, last_word_last_synset]
    except (TypeError, KeyError, IndexError):
        # return large value if word not in wordnet
        return [100, 100, 100]


results_s2v_verb = open("D:\\Results\\s2v_ran_verbs_results.csv", "w", newline='')
fieldnames = ['verb', 'first word in first synset', 'last word in first synset', 'last word in last synset']
results_writer = csv.DictWriter(results_s2v_verb, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
results_writer.writeheader()
for i in range(1000):
    verb = random.choice(verbs)
    print(verb)
    try:
        wordnet_verbs = wordnet_first_last(verb, wn.VERB)
    except KeyError:
        continue
    try:
        first_first = s2v.similarity([verb + '|VERB'], [wordnet_verbs[0] + '|VERB'])
        last_first = s2v.similarity([verb + '|VERB'], [wordnet_verbs[1] + '|VERB'])
        last_last = s2v.similarity([verb + '|VERB'], [wordnet_verbs[1] + '|VERB'])
        if first_first <= 1 and last_first <= 1 and last_last <= 1:
            results_writer.writerow({'verb': verb,
                                     'first word in first synset': first_first,
                                     'last word in first synset': last_first,
                                     'last word in last synset': last_last})
    except (TypeError, KeyError):
        # ignore phrasal verbs, only single verbs accepted by s2v
        continue

results_s2v_verb.close()

results_s2v_noun = open("D:\\Results\\ran_nouns_results.csv", "w", newline='')
fieldnames = ['noun', 'first word in first synset', 'last word in first synset', 'last word in last synset']
results_writer = csv.DictWriter(results_s2v_noun, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
results_writer.writeheader()
for i in range(10000):
    noun = random.choice(nouns)
    print(noun)
    try:
        wordnet_verbs = wordnet_first_last(noun, wn.NOUN)
    except KeyError:
        continue
    try:
        first_first = s2v.similarity([noun + '|NOUN'], [wordnet_verbs[0] + '|NOUN'])
        last_first = s2v.similarity([noun + '|NOUN'], [wordnet_verbs[1] + '|NOUN'])
        last_last = s2v.similarity([noun + '|NOUN'], [wordnet_verbs[1] + '|NOUN'])
        if first_first <= 1 and last_first <= 1 and last_last <= 1:
            results_writer.writerow({'noun': noun,
                                     'first word in first synset': first_first,
                                     'last word in first synset': last_first,
                                     'last word in last synset': last_last})
    except (TypeError, KeyError):
        # ignore phrasal verbs, only single verbs accepted by s2v
        continue

results_s2v_noun.close()

results_w2v_verbs = open("D:\\Results\\ran_w2v_verbs_results.csv", "w", newline='')
fieldnames = ['verb', 'first word in first synset', 'last word in first synset', 'last word in last synset']
results_writer = csv.DictWriter(results_w2v_verbs, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
results_writer.writeheader()
for i in range(1000):
    verb = random.choice(verbs)
    print(verb)
    try:
        wordnet_verbs = wordnet_first_last(verb, wn.VERB)
    except KeyError:
        continue
    try:
        first_first = w2v.similarity(verb, wordnet_verbs[0])
        last_first = w2v.similarity(verb, wordnet_verbs[1])
        last_last = w2v.similarity(verb, wordnet_verbs[2])
        if first_first <= 1 and last_first <= 1 and last_last <= 1:
            results_writer.writerow({'verb': verb,
                                     'first word in first synset': first_first,
                                     'last word in first synset': last_first,
                                     'last word in last synset': last_last})
    except (TypeError, KeyError):
        # ignore phrasal verbs, only single verbs accepted by w2v
        continue

results_w2v_verbs.close()

results_w2v_noun = open("D:\\Results\\ran_w2v_nouns_results.csv", "w", newline='')
fieldnames = ['noun', 'first word in first synset', 'last word in first synset', 'last word in last synset']
results_writer = csv.DictWriter(results_w2v_noun, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
results_writer.writeheader()
for i in range(10000):
    noun = random.choice(nouns)
    print(noun)
    try:
        wordnet_verbs = wordnet_first_last(noun, wn.NOUN)
    except KeyError:
        continue
    try:
        first_first = w2v.similarity(noun, wordnet_verbs[0])
        last_first = w2v.similarity(noun, wordnet_verbs[1])
        last_last = w2v.similarity(noun, wordnet_verbs[2])
        if first_first <= 1 and last_first <= 1 and last_last <= 1:
            results_writer.writerow({'noun': noun,
                                     'first word in first synset': first_first,
                                     'last word in first synset': last_first,
                                     'last word in last synset': last_last})
    except (TypeError, KeyError):
        # ignore phrasal verbs, only single verbs accepted by w2v
        continue

results_w2v_noun.close()

# sanity check
# for i in range(1):
#     noun = "wall"
#     print(noun)
#     synsets = wn.synsets(noun, pos=wn.NOUN)
#     print(synsets)
#     first_synset = synsets[0]
#     first_synset = re.search("'.*'", str(first_synset)).group(0)
#     print(first_synset)
#     last_synset = synsets[-1]
#     last_synset = re.search("'.*'", str(last_synset)).group(0)
#     first_synset = [str(lemma.name()) for lemma in wn.synset(first_synset.replace("'", "")).lemmas()]
#     last_synset = [str(lemma.name()) for lemma in wn.synset(last_synset.replace("'", "")).lemmas()]
#     print(first_synset)
#     print(last_synset)
#     first_word_first_synset = first_synset[0]
#     last_word_first_synset = first_synset[-1]
#     last_word_last_synset = last_synset[-1]
#     print(first_word_first_synset)
#     print(last_word_last_synset)
#     print(last_word_first_synset)

