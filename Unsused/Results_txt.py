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


results = open("E:\\Results\\results.txt", "w")
w2v_model = "E:\\WikiData\\Trained\\word2vec\\10GB\\word2vec.model"
s2v_model = "E:\\WikiData\\Trained\\sense2vec\\RedditVecs\\sense2vec-vectors"

w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

########################################################################################################################

results.write("sense2vec")
results.write("Word lists")

########################################################################################################################
results.write("word2vec")
results.write("Word lists")
write_string = ""
write_string += ("2similarity('walked', 'talk') " + str(w2v.similarity('walked', 'talk')))
write_string += ("2similarity('walked', 'slept') " + str(w2v.similarity('walked', 'slept')))
write_string += ("2model2.similarity('run', 'walk') " + str(w2v.similarity('run', 'walk')))
write_string += ("2model2.similarity('run', 'drive') " + str(w2v.similarity('run', 'drive')))
write_string += ("2model2.similarity('run', 'talk') " + str(w2v.similarity('run', 'talk')))
write_string += ("2model2.similarity('run', 'hatter') " + str(w2v.similarity('run', 'hatter')))
write_string += ("2model2.similarity('run', 'think') " + str(w2v.similarity('run', 'think')))

write_string += "Most similar"
write_string += ("Most similar 'run' " + str(word_vec_query("run")))
write_string += ("Most similar 'walked' " + str(word_vec_query("walked")))
write_string += ("Most similar 'jogged' " + str(word_vec_query("jogged")))
write_string += ("Most similar 'talked' " + str(word_vec_query("talked")))
write_string += ("Most similar 'slept' " + str(word_vec_query("slept")))

write_string += "Logical entailment"
write_string += "Compositions"
write_string += ("'take' + 'pay'" + str(word_vec_query(None, ["take", "pay"])))

write_string += "Opposites"
write_string += ("'buy' - 'sell'" + str(word_vec_query(None, ["buy"], ["sell"])))
write_string += ("'get' - 'give'" + str(word_vec_query(None, ["get"], ["give"])))
write_string += "Difference: 'buy' - 'sell' ~ 'get' - 'give'"
# write_string += str((word_vec_query(None, ["buy"], ["sell"])) - (word_vec_query(None, ["get"], ["give"])))

########################################################################################################################

results.write("sense2vec")
results.write("Word lists")
write_string = ""
write_string += ("2similarity('walked', 'talk') " + str(s2v.similarity(['walked' + '|VERB'], ['talk' + '|VERB'])))
write_string += ("2similarity('walked', 'slept') " + str(s2v.similarity(['walked' + '|VERB'], ['slept' + '|VERB'])))
write_string += ("2model2.similarity('run', 'walk') " + str(s2v.similarity(['run' + '|VERB'], ['walk' + '|VERB'])))
write_string += ("2model2.similarity('run', 'drive') " + str(s2v.similarity(['run' + '|VERB'], ['drive' + '|VERB'])))
write_string += ("2model2.similarity('run', 'talk') " + str(s2v.similarity(['run' + '|VERB'], ['talk' + '|VERB'])))
write_string += ("2model2.similarity('run', 'hatter') " + str(s2v.similarity(['run' + '|VERB'], ['hatter' + '|NOUN'])))
write_string += ("2model2.similarity('run', 'think') " + str(s2v.similarity(['run' + '|VERB'], ['think' + '|VERB'])))
results.write(str(write_string))

results.write("Most similar")
write_string = ""
write_string += ("Most similar 'run' " + str(sense_vec_query("run" + '|VERB')))
write_string += ("Most similar 'walked' " + str(sense_vec_query("walked" + '|VERB')))
write_string += ("Most similar 'jogged' " + str(sense_vec_query("jogged" + '|VERB')))
write_string += ("Most similar 'talked' " + str(sense_vec_query("talked" + '|VERB')))
write_string += ("Most similar 'slept' " + str(sense_vec_query("slept" + '|VERB')))
results.write(str(write_string))

results.write("Logical entailment")
results.write("Compositions")

results.write("Opposites")

########################################################################################################################
