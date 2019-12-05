import gensim
from sense2vec import Sense2Vec


def word_vec_query(verb, *args, **compositions):
    if args:
        positive = []
        negative = []
        for key in compositions:
            if key == 'p':
                positive = key
            if key == 'n':
                negative = key
        return w2v.most_similar(positive, negative)
    return w2v.most_similar(verb)


def sense_vec_query(verb):
    query = verb
    assert query in s2v
    # vector = s2v[query]
    # freq = s2v.get_freq(query)
    return s2v.most_similar(query, n=3)


results = open("D:\Results\\results.txt", "w")
w2v_model = "D:\WikiData\Trained\word2vec\\10GB\word2vec.model"
s2v_model = "D:\WikiData\Trained\sense2vec\RedditVecs\sense2vec-vectors"

w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

write_string = ""

########################################################################################################################

results.write("sense2vec")
results.write("Word lists")

########################################################################################################################
results.write("word2vec")
results.write("Word lists")
write_string += ("2similarity('walked', 'talk') " + str(w2v.similarity('walked', 'talk')))
write_string += ("2similarity('walked', 'slept') " + str(w2v.similarity('walked', 'slept')))
write_string += ("2model2.similarity('run', 'walk') " + str(w2v.similarity('run', 'walk')))
write_string += ("2model2.similarity('run', 'drive') " + str(w2v.similarity('run', 'drive')))
write_string += ("2model2.similarity('run', 'talk') " + str(w2v.similarity('run', 'talk')))
write_string += ("2model2.similarity('run', 'hatter') " + str(w2v.similarity('run', 'hatter')))
write_string += ("2model2.similarity('run', 'think') " + str(w2v.similarity('run', 'think')))
results.wite(str(write_string))

results.write("Most similar")
results.write("Most similar 'run' ", word_vec_query("run"))
results.write("Most similar 'walked' ", word_vec_query("walked"))
results.write("Most similar 'jogged' ", word_vec_query("jogged"))
results.write("Most similar 'talked' ", word_vec_query("talked"))
results.write("Most similar 'slept' ", word_vec_query("slept"))

results.write("Logical entailment")
results.write("Compositions")
results.write("'take' + 'pay'", word_vec_query(None, p=["take", "pay"]))

results.write("Opposites")
results.write("'buy' - 'sell'", word_vec_query(None, p=["buy"], n=["sell"]))
results.write("'get' - 'give'", word_vec_query(None, p=["get"], n=["give"]))
results.write("Difference: 'buy' - 'sell' ~ 'get' - 'give'")
results.write(word_vec_query(None, p=["buy"], n=["sell"])[0] - word_vec_query(None, p=["get"], n=["give"])[0])

########################################################################################################################

results.write("sense2vec")
results.write("Word lists")
results.write("2similarity('walked', 'talk') ", s2v.similarity('walked', 'talk'))
results.write("2similarity('walked', 'slept') ", s2v.similarity('walked', 'slept'))
results.write("2model2.similarity('run', 'walk')", s2v.similarity('run', 'walk'))
results.write("2model2.similarity('run', 'drive')", s2v.similarity('run', 'drive'))
results.write("2model2.similarity('run', 'talk')", s2v.similarity('run', 'talk'))
results.write("2model2.similarity('run', 'hatter')", s2v.similarity('run', 'hatter'))
results.write("2model2.similarity('run', 'think')", s2v.similarity('run', 'think'))

results.write("Most similar")
results.write("Most similar 'run' ", sense_vec_query("run"))
results.write("Most similar 'walked' ", sense_vec_query("walked"))
results.write("Most similar 'jogged' ", sense_vec_query("jogged"))
results.write("Most similar 'talked' ", sense_vec_query("talked"))
results.write("Most similar 'slept' ", sense_vec_query("slept"))

results.write("Logical entailment")
results.write("Compositions")

results.write("Opposites")

########################################################################################################################
