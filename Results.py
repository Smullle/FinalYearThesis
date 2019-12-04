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


def sense_query(verb):
    query = verb
    assert query in s2v
    # vector = s2v[query]
    # freq = s2v.get_freq(query)
    return s2v.most_similar(query, n=3)


results = open("D:\Results\results.txt", "w")
w2v_model = "D:\GoogleNews-vectors-negative300.bin"
s2v_model = "D:\Wiki\Trained\sense2vec\sense2vec"

w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_model, binary=True)
s2v = Sense2Vec().from_disk(s2v_model)

########################################################################################################################

results.write("sense2vec")
results.write("Word lists")

########################################################################################################################
results.write("word2vec")
results.write("Word lists")
results.write("2similarity('walked', 'talk') ", w2v.similarity('walked', 'talk'))
results.write("2similarity('walked', 'slept') ", w2v.similarity('walked', 'slept'))
results.write("2model2.similarity('run', 'walk')", w2v.similarity('run', 'walk'))
results.write("2model2.similarity('run', 'drive')", w2v.similarity('run', 'drive'))
results.write("2model2.similarity('run', 'talk')", w2v.similarity('run', 'talk'))
results.write("2model2.similarity('run', 'hatter')", w2v.similarity('run', 'hatter'))
results.write("2model2.similarity('run', 'think')", w2v.similarity('run', 'think'))

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

results.write("Logical entailment")
results.write("Compositions")

results.write("Opposites")

########################################################################################################################
