import csv
import gensim
from sense2vec import Sense2Vec


def word_vec_query(verb, positive=None, negative=None, topn=1):
    if verb is None and positive is None:
        return w2v.most_similar(negative, topn=topn)
    elif verb is None and negative is None:
        return w2v.most_similar(positive, topn=topn)
    elif verb is None:
        return w2v.most_similar(positive, negative, topn=topn)
    return w2v.most_similar(verb, topn=topn)


def sense_vec_query(verb, n=3):
    query = verb
    assert query in s2v
    # vector = s2v[query]
    # freq = s2v.get_freq(query)
    return s2v.most_similar(query, n=n)


results = open("D:\\Results\\results.csv", "a")


results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
w2v_model = "D:\\WikiData\\Trained\\word2vec\\10GB\\word2vec.model"
s2v_model = "D:\\WikiData\\Trained\\sense2vec\\RedditVecs\\sense2vec-vectors"

w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

########################################################################################################################
fieldnames = ['Future Past', 'word2vec', 'sense2vec', 'WordNet']

results_writer.writerow({'Future-Past': 'run-ran',
                         'word2vec': w2v.similarity('run', 'ran'),
                         'sense2vec': s2v.similarity(['run' + '|VERB'], ['ran' + '|VERB']),
                         'WordNet': ''})

########################################################################################################################
fieldnames = ['Most Similar', 'word2vec', 'sense2vec']

results_writer.writerow({'Most Similar': 'run',
                         'word2vec': word_vec_query("run"),
                         'sense2vec': sense_vec_query("run" + '|VERB')})

# write_string += "Most similar"
# write_string += ("Most similar 'run' " + str(word_vec_query("run")))
# write_string += ("Most similar 'walked' " + str(word_vec_query("walked")))
# write_string += ("Most similar 'jogged' " + str(word_vec_query("jogged")))
# write_string += ("Most similar 'talked' " + str(word_vec_query("talked")))
# write_string += ("Most similar 'slept' " + str(word_vec_query("slept")))

########################################################################################################################
fieldnames = ['Oddballs', 'word2vec', 'sense2vec-VERB', 'sense2vec-NOUN']

results_writer.writerow({'Oddballs': 'sleep',
                         'word2vec': word_vec_query("sleep"),
                         'sense2vec-VERB': sense_vec_query("run" + '|VERB'),
                         'sense2vec-NOUN': sense_vec_query("run" + '|NOUN')})

########################################################################################################################
fieldnames = ['Verb to Verb', 'word2vec', 'sense2vec']

results_writer.writerow({'Verb to Verb': 'run-walk',
                         'word2vec': w2v.similarity('run', 'walk'),
                         'sense2vec': s2v.similarity(['run' + '|VERB'], ['walk' + '|VERB']),
                         'WordNet': ''})

# write_string += ("2similarity('walked', 'talk') " + str(w2v.similarity('walked', 'talk')))
# write_string += ("2similarity('walked', 'slept') " + str(w2v.similarity('walked', 'slept')))
# write_string += ("2model2.similarity('run', 'walk') " + str(w2v.similarity('run', 'walk')))
# write_string += ("2model2.similarity('run', 'drive') " + str(w2v.similarity('run', 'drive')))
# write_string += ("2model2.similarity('run', 'talk') " + str(w2v.similarity('run', 'talk')))
# write_string += ("2model2.similarity('run', 'hatter') " + str(w2v.similarity('run', 'hatter')))
# write_string += ("2model2.similarity('run', 'think') " + str(w2v.similarity('run', 'think')))

########################################################################################################################
fieldnames = ['Noun to Noun', 'word2vec', 'sense2vec']

results_writer.writerow({'Noun to Noun': 'Laptop-Music',
                         'word2vec': w2v.similarity('Laptop', 'Music'),
                         'sense2vec': s2v.similarity(['Laptop' + '|NOUN'], ['Music' + '|NOUN']),
                         'WordNet': ''})

########################################################################################################################
fieldnames = ['Preposition to Preposition', 'word2vec', 'sense2vec']

results_writer.writerow({'Noun to Noun': 'in-of',
                         'word2vec': w2v.similarity('in', 'of'),
                         'sense2vec': s2v.similarity(['in' + '|ADP'], ['of' + '|ADP']),
                         'WordNet': ''})

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
