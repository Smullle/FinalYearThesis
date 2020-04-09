import csv
import gensim
from sense2vec import Sense2Vec
from nltk.corpus import wordnet as wn
import re
from nltk.corpus import words
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def word_vec_query(verb, positive=None, negative=None, topn=1):
    if verb is None and positive is None:
        return w2v.most_similar(negative, topn=topn)
    elif verb is None and negative is None:
        return w2v.most_similar(positive, topn=topn)
    elif verb is None:
        return w2v.most_similar(positive, negative, topn=topn)
    return w2v.most_similar(verb, topn=topn)


def sense_vec_query(verb, n=500):
    query = verb
    assert query in s2v
    # vector = s2v[query]
    # freq = s2v.get_freq(query)
    word_list = s2v.most_similar(query, n=n)
    ans = []
    for word in word_list:
        word = str(word)
        # print(word[2:].split('|')[0])
        z = re.match("^[a-zA-Z]+$", word[2:].split('|')[0])
        if z and word[2:].split('|')[0] in words.words():
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

w2v = gensim.models.Word2Vec.load(w2v_model)
print("Word2Vec Model Loaded")
s2v = Sense2Vec().from_disk(s2v_model)
print("Sense2Vec Model Loaded")

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

# write_string += "Most similar"
# write_string += ("Most similar 'run' " + str(word_vec_query("run")))
# write_string += ("Most similar 'walked' " + str(word_vec_query("walked")))
# write_string += ("Most similar 'jogged' " + str(word_vec_query("jogged")))
# write_string += ("Most similar 'talked' " + str(word_vec_query("talked")))
# write_string += ("Most similar 'slept' " + str(word_vec_query("slept")))

print("Most Similar Complete")
########################################################################################################################
fieldnames = ['Collisions', 'word2vec', 'sense2vec-VERB', 'sense2vec-NOUN']
results_writer = csv.DictWriter(results, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)

results_writer.writeheader()
results_writer.writerow({'Collisions': 'damage',
                         'word2vec': word_vec_query("damage", topn=6),
                         'sense2vec-VERB': sense_vec_query("damage" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("damage" + '|NOUN')})
results_writer.writerow({'Collisions': 'play',
                         'word2vec': word_vec_query("play", topn=6),
                         'sense2vec-VERB': sense_vec_query("play" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("play" + '|NOUN')})
results_writer.writerow({'Collisions': 'work',
                         'word2vec': word_vec_query("work", topn=6),
                         'sense2vec-VERB': sense_vec_query("work" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("work" + '|NOUN')})
results_writer.writerow({'Collisions': 'sink',
                         'word2vec': word_vec_query("sink", topn=6),
                         'sense2vec-VERB': sense_vec_query("sink" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("sink" + '|NOUN')})
results_writer.writerow({'Collisions': 'fire',
                         'word2vec': word_vec_query("fire", topn=6),
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

# write_string += ("2similarity('walked', 'talk') " + str(w2v.similarity('walked', 'talk')))
# write_string += ("2similarity('walked', 'slept') " + str(w2v.similarity('walked', 'slept')))
# write_string += ("2model2.similarity('run', 'walk') " + str(w2v.similarity('run', 'walk')))
# write_string += ("2model2.similarity('run', 'drive') " + str(w2v.similarity('run', 'drive')))
# write_string += ("2model2.similarity('run', 'talk') " + str(w2v.similarity('run', 'talk')))
# write_string += ("2model2.similarity('run', 'hatter') " + str(w2v.similarity('run', 'hatter')))
# write_string += ("2model2.similarity('run', 'think') " + str(w2v.similarity('run', 'think')))

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
########################################################################################################################

# write_string += "Logical entailment"
# write_string += "Compositions"
# write_string += ("'take' + 'pay'" + str(word_vec_query(None, ["take", "pay"])))
#
# write_string += "Opposites"
# write_string += ("'buy' - 'sell'" + str(word_vec_query(None, ["buy"], ["sell"])))
# write_string += ("'get' - 'give'" + str(word_vec_query(None, ["get"], ["give"])))
# write_string += "Difference: 'buy' - 'sell' ~ 'get' - 'give'"
# # write_string += str((word_vec_query(None, ["buy"], ["sell"])) - (word_vec_query(None, ["get"], ["give"])))
#
#
#
# results.write("Logical entailment")
# results.write("Compositions")
#
# results.write("Opposites")

########################################################################################################################
