import csv
import gensim
from sense2vec import Sense2Vec
from nltk.corpus import wordnet as wn
import re
from nltk.corpus import words
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def word_vec_query(verb, positive=None, negative=None, topn=1):
    """
    Function to obtain the most similar vectors for a given word, number of vectors returned depends on topn value.
    Optional: add or subtract vectors to or from the supplied word using positive or negative args.
    :param verb: input of single word to return a list of topn most similar vectors
    :param positive: (optional) word to add to verb vector
    :param negative: (optional) word to subtract from the verb vectors
    :param topn: (default = 1) number of cosine similar vectors to return
    :return: List of topn most similar vectors (tuple of word and similarity value)
    """
    if verb is None and positive is None:
        return w2v.most_similar(negative, topn=topn)
    elif verb is None and negative is None:
        return w2v.most_similar(positive, topn=topn)
    elif verb is None:
        return w2v.most_similar(positive, negative, topn=topn)
    word_list = w2v.most_similar(verb, topn=topn)
    ans = []
    # only return the words not similarity score, required for cleaner results
    # remove to return a list of tuples
    for word in word_list:
        word = str(word)
        word = word[2:].split(',')[0]
        ans.append(word)
    return ans


def sense_vec_query(verb, n=500):
    """
    Function to obtain the most similar vectors for a given POS TAGGED word.
    Number of vectors returned depends on n value, multi-words removed eg. words containing "-" "_" etc.
    :param verb: input of single POS TAGGED word to return a list of n most similar vectors
    :param n: (default = 500) number of cosine similar vectors to return
    :return: List of n most similar vectors (tuple of tagged word and similarity value)
    """
    query = verb
    assert query in s2v
    word_list = s2v.most_similar(query, n=n)
    ans = []
    for word in word_list:
        word = str(word)
        word = word[2:].split('|')[0]  # only return the word without POS tag and similarity score
        z = re.match("^[a-zA-Z]+$", word)  # remove multi-words
        if z and word in words.words():  # Use nltk dictionary to filter non english words.
            ans.append(word)
    return ans


def wordnet_similarity(word1, word2, pos):
    word1 = wn.synsets(word1, pos=pos)[0]
    word2 = wn.synsets(word2, pos=pos)[0]
    word1 = re.search("'.*'", str(word1)).group(0)
    word2 = re.search("'.*'", str(word2)).group(0)
    word1 = wn.synset(word1.replace("'", ""))
    word2 = wn.synset(word2.replace("'", ""))
    return word1.path_similarity(word2)


w2v_model = "/home/shanesmullen/train/vmshare/models/word2vec/word2vec.model"
s2v_model = "/home/shanesmullen/train/vmshare/models/sense2vec"
results_file = "/home/shanesmullen/train/vmshare/results/results.csv"

try:
    w2v = gensim.models.Word2Vec.load(w2v_model)
    print("Word2Vec Model Loaded")
    s2v = Sense2Vec().from_disk(s2v_model)
    print("Sense2Vec Model Loaded")
except FileNotFoundError:
    print("Incorrect model location: ", w2v_model, " or ", s2v_model)

results = open(results_file, "w", newline='')
print("Results File Created at:", results_file)

########################################################################################################################
fieldnames = ['Future-Past', 'word2vec', 'sense2vec', 'WordNet']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Future-Past': 'run - ran',
                         'word2vec': w2v.similarity('run', 'ran'),
                         'sense2vec': s2v.similarity(['run' + '|VERB'], ['ran' + '|VERB']),
                         'WordNet': wordnet_similarity('run', 'ran', wn.VERB)})
results_writer.writerow({'Future-Past': 'talk - talked',
                         'word2vec': w2v.similarity('talk', 'talked'),
                         'sense2vec': s2v.similarity(['talk' + '|VERB'], ['talked' + '|VERB']),
                         'WordNet': wordnet_similarity('talk', 'talked', wn.VERB)})
results_writer.writerow({'Future-Past': 'see - seen',
                         'word2vec': w2v.similarity('see', 'seen'),
                         'sense2vec': s2v.similarity(['see' + '|VERB'], ['seen' + '|VERB']),
                         'WordNet': wordnet_similarity('see', 'seen', wn.VERB)})
results_writer.writerow({'Future-Past': 'have - had',
                         'word2vec': w2v.similarity('have', 'had'),
                         'sense2vec': s2v.similarity(['have' + '|VERB'], ['had' + '|VERB']),
                         'WordNet': wordnet_similarity('have', 'had', wn.VERB)})
results_writer.writerow({'Future-Past': 'hold - held',
                         'word2vec': w2v.similarity('hold', 'held'),
                         'sense2vec': s2v.similarity(['hold' + '|VERB'], ['held' + '|VERB']),
                         'WordNet': wordnet_similarity('hold', 'held', wn.VERB)})
results_writer.writerow({'Future-Past': 'say - said',
                         'word2vec': w2v.similarity('say', 'said'),
                         'sense2vec': s2v.similarity(['say' + '|VERB'], ['said' + '|VERB']),
                         'WordNet': wordnet_similarity('say', 'said', wn.VERB)})
results_writer.writerow({'Future-Past': 'lose - lost',
                         'word2vec': w2v.similarity('lose', 'lost'),
                         'sense2vec': s2v.similarity(['lose' + '|VERB'], ['lost' + '|VERB']),
                         'WordNet': wordnet_similarity('lose', 'lost', wn.VERB)})

print("Future-Past Complete")
########################################################################################################################
fieldnames = ['Most Similar', 'word2vec', 'sense2vec']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Most Similar': 'run',
                         'word2vec': word_vec_query("run", topn=6),
                         'sense2vec': sense_vec_query("run" + '|VERB')})
results_writer.writerow({'Most Similar': '',
                         'word2vec': '',
                         'sense2vec': ''})

print("Most Similar Complete")
########################################################################################################################
fieldnames = ['Collisions', 'word2vec', 'sense2vec-VERB', 'sense2vec-NOUN']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Collisions': 'water',
                         'word2vec': word_vec_query("water", topn=10),
                         'sense2vec-VERB': sense_vec_query("water" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("water" + '|NOUN')})
results_writer.writerow({'Collisions': 'play',
                         'word2vec': word_vec_query("play", topn=10),
                         'sense2vec-VERB': sense_vec_query("play" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("play" + '|NOUN')})
results_writer.writerow({'Collisions': 'work',
                         'word2vec': word_vec_query("work", topn=10),
                         'sense2vec-VERB': sense_vec_query("work" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("work" + '|NOUN')})
results_writer.writerow({'Collisions': 'sink',
                         'word2vec': word_vec_query("sink", topn=10),
                         'sense2vec-VERB': sense_vec_query("sink" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("sink" + '|NOUN')})
results_writer.writerow({'Collisions': 'fire',
                         'word2vec': word_vec_query("fire", topn=10),
                         'sense2vec-VERB': sense_vec_query("fire" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("fire" + '|NOUN')})


print("Collisions Complete")
########################################################################################################################
fieldnames = ['Verb to Verb', 'word2vec', 'sense2vec', 'WordNet']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Verb to Verb': 'lose - win',
                         'word2vec': w2v.similarity('lose', 'win'),
                         'sense2vec': s2v.similarity(['lose' + '|VERB'], ['win' + '|VERB']),
                         'WordNet': wordnet_similarity('lose', 'win', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'shout - whisper',
                         'word2vec': w2v.similarity('shout', 'whisper'),
                         'sense2vec': s2v.similarity(['shout' + '|VERB'], ['whisper' + '|VERB']),
                         'WordNet': wordnet_similarity('shout', 'whisper', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'float - sink',
                         'word2vec': w2v.similarity('float', 'sink'),
                         'sense2vec': s2v.similarity(['float' + '|VERB'], ['sink' + '|VERB']),
                         'WordNet': wordnet_similarity('float', 'sink', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'borrow - lend',
                         'word2vec': w2v.similarity('borrow', 'lend'),
                         'sense2vec': s2v.similarity(['borrow' + '|VERB'], ['lend' + '|VERB']),
                         'WordNet': wordnet_similarity('borrow', 'lend', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'build - destroy',
                         'word2vec': w2v.similarity('build', 'destroy'),
                         'sense2vec': s2v.similarity(['build' + '|VERB'], ['destroy' + '|VERB']),
                         'WordNet': wordnet_similarity('build', 'destroy', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'punish - reward',
                         'word2vec': w2v.similarity('punish', 'reward'),
                         'sense2vec': s2v.similarity(['punish' + '|VERB'], ['reward' + '|VERB']),
                         'WordNet': wordnet_similarity('punish', 'reward', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'show - hide',
                         'word2vec': w2v.similarity('show', 'hide'),
                         'sense2vec': s2v.similarity(['show' + '|VERB'], ['hide' + '|VERB']),
                         'WordNet': wordnet_similarity('show', 'hide', wn.VERB)})
results_writer.writerow({'Verb to Verb': 'laugh - cry',
                         'word2vec': w2v.similarity('laugh', 'cry'),
                         'sense2vec': s2v.similarity(['laugh' + '|VERB'], ['cry' + '|VERB']),
                         'WordNet': wordnet_similarity('laugh', 'cry', wn.VERB)})

print("Verb to Verb Complete")
########################################################################################################################
fieldnames = ['Noun to Noun', 'word2vec', 'sense2vec', 'WordNet']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Noun to Noun': 'laptop - music',
                         'word2vec': w2v.similarity('laptop', 'music'),
                         'sense2vec': s2v.similarity(['laptop' + '|NOUN'], ['music' + '|NOUN']),
                         'WordNet': wordnet_similarity('laptop', 'music', wn.NOUN)})
results_writer.writerow({'Noun to Noun': 'car - drive',
                         'word2vec': w2v.similarity('car', 'drive'),
                         'sense2vec': s2v.similarity(['car' + '|NOUN'], ['drive' + '|NOUN']),
                         'WordNet': wordnet_similarity('car', 'drive', wn.NOUN)})
results_writer.writerow({'Noun to Noun': 'window - house',
                         'word2vec': w2v.similarity('window', 'house'),
                         'sense2vec': s2v.similarity(['window' + '|NOUN'], ['house' + '|NOUN']),
                         'WordNet': wordnet_similarity('window', 'house', wn.NOUN)})
results_writer.writerow({'Noun to Noun': 'actor - play',
                         'word2vec': w2v.similarity('actor', 'play'),
                         'sense2vec': s2v.similarity(['actor' + '|NOUN'], ['play' + '|NOUN']),
                         'WordNet': wordnet_similarity('actor', 'play', wn.NOUN)})
results_writer.writerow({'Noun to Noun': 'fire - water',
                         'word2vec': w2v.similarity('fire', 'water'),
                         'sense2vec': s2v.similarity(['fire' + '|NOUN'], ['water' + '|NOUN']),
                         'WordNet': wordnet_similarity('fire', 'water', wn.NOUN)})
results_writer.writerow({'Noun to Noun': 'work - pay',
                         'word2vec': w2v.similarity('work', 'pay'),
                         'sense2vec': s2v.similarity(['work' + '|NOUN'], ['pay' + '|NOUN']),
                         'WordNet': wordnet_similarity('work', 'pay', wn.NOUN)})

print("Noun to Noun Complete")
########################################################################################################################
fieldnames = ['Preposition to Preposition', 'word2vec', 'sense2vec']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Preposition to Preposition': 'above - below',
                         'word2vec': w2v.similarity('above', 'below'),
                         'sense2vec': s2v.similarity(['above' + '|ADP'], ['below' + '|ADP'])})
results_writer.writerow({'Preposition to Preposition': 'inside - outside',
                         'word2vec': w2v.similarity('inside', 'outside'),
                         'sense2vec': s2v.similarity(['inside' + '|ADP'], ['outside' + '|ADP'])})
results_writer.writerow({'Preposition to Preposition': 'with - without',
                         'word2vec': w2v.similarity('above', 'without'),
                         'sense2vec': s2v.similarity(['above' + '|ADP'], ['without' + '|ADP'])})
results_writer.writerow({'Preposition to Preposition': 'up - down',
                         'word2vec': w2v.similarity('up', 'down'),
                         'sense2vec': s2v.similarity(['up' + '|ADP'], ['down' + '|ADP'])})
results_writer.writerow({'Preposition to Preposition': 'before - after',
                         'word2vec': w2v.similarity('before', 'after'),
                         'sense2vec': s2v.similarity(['before' + '|ADP'], ['after' + '|ADP'])})
# results_writer.writerow({'Preposition to Preposition': 'far - close',
#                          'word2vec': w2v.similarity('far', 'close'),
#                          'sense2vec': s2v.similarity(['far' + '|ADP'], ['close' + '|ADP'])})
results_writer.writerow({'Preposition to Preposition': 'against - for',
                         'word2vec': w2v.similarity('against', 'for'),
                         'sense2vec': s2v.similarity(['against' + '|ADP'], ['for' + '|ADP'])})

print("Preposition to Preposition Complete")
