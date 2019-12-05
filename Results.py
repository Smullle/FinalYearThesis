import gensim
from sense2vec import Sense2Vec


def word_vec_query(verb, topn, *args, **compositions):
    if args:
        positive = []
        negative = []
        for key in compositions:
            if key == 'p':
                positive = key
            if key == 'n':
                negative = key
        return w2v.most_similar(positive, negative)
    return w2v.most_similar(verb, topn)


def sense_vec_query(verb):
    query = verb
    assert query in s2v
    # vector = s2v[query]
    # freq = s2v.get_freq(query)
    return s2v.most_similar(query, n=3)


results = open("X:\Thesis\Results\\results.txt", "w")
w2v_model = "X:\Thesis\word2vec.model"
s2v_model = "X:\Thesis\sense2vec-vectors"

w2v = gensim.models.Word2Vec.load(w2v_model)
s2v = Sense2Vec().from_disk(s2v_model)

########################################################################################################################

results.write("sense2vec" + "\n")
results.write("Word lists" + "\n")

########################################################################################################################
results.write("word2vec" + "\n")
results.write("Word lists" + "\n")
write_string = ""
write_string += ("Similarity('walked', 'talk') " + str(w2v.similarity('walked', 'talk'))) + "\n"
write_string += ("Similarity('walked', 'slept') " + str(w2v.similarity('walked', 'slept'))) + "\n"
write_string += ("Similarity('run', 'walk') " + str(w2v.similarity('run', 'walk'))) + "\n"
write_string += ("Similarity('run', 'drive') " + str(w2v.similarity('run', 'drive'))) + "\n"
write_string += ("Similarity('run', 'talk') " + str(w2v.similarity('run', 'talk'))) + "\n"
write_string += ("Similarity('run', 'hatter') " + str(w2v.similarity('run', 'hatter'))) + "\n"
write_string += ("Similarity('run', 'think') " + str(w2v.similarity('run', 'think'))) + "\n"

write_string += ("Most similar" + "\n")
write_string += ("Most similar 'run' " + str(w2v.most_similar("run", topn=3))) + "\n"
write_string += ("Most similar 'walked' " + str(w2v.most_similar("walked", topn=3))) + "\n"
write_string += ("Most similar 'jogged' " + str(w2v.most_similar("jogged", topn=3))) + "\n"
write_string += ("Most similar 'talked' " + str(w2v.most_similar("talked", topn=3))) + "\n"
write_string += ("Most similar 'slept' " + str(w2v.most_similar("slept", topn=3))) + "\n"

write_string += ("Logical entailment" + "\n")
write_string += ("Compositions" + "\n")
write_string += ("'take' + 'pay'" + str(w2v.most_similar(["take", "pay"], topn=3))) + "\n"

write_string += ("Opposites" + "\n")
write_string += ("'buy' - 'sell'" + str(w2v.most_similar(["buy"], ["sell"], topn=3))) + "\n"
write_string += ("'get' - 'give'" + str(w2v.most_similar(["get"], ["give"], topn=3))) + "\n"
#write_string += ("Difference: 'buy' - 'sell' ~ 'get' - 'give'")
#write_string +=  + str((word_vec_query(None, p=["buy"], n=["sell"])[0] - word_vec_query(None, p=["get"], n=["give"])[0]))
results.write(str(write_string))

########################################################################################################################

results.write("sense2vec" + "\n")
results.write("Word lists" + "\n")
write_string = ""
write_string += ("Similarity('walked', 'talk') " + str(s2v.similarity('walked|VERB', 'talk|VERB'))) + "\n"
write_string += ("Similarity('walked', 'slept') " + str(s2v.similarity('walked|VERB', 'slept|VERB'))) + "\n"
write_string += ("Similarity('run', 'walk') " + str(s2v.similarity('run|VERB', 'walk|VERB'))) + "\n"
write_string += ("Similarity('run', 'drive') " + str(s2v.similarity('run|VERB', 'drive|VERB'))) + "\n"
write_string += ("Similarity('run', 'talk') " + str(s2v.similarity('run|VERB', 'talk|VERB'))) + "\n"
write_string += ("Similarity('run', 'hatter') " + str(s2v.similarity('run|VERB', 'hatter|NOUN'))) + "\n"
write_string += ("Similarity('run', 'think') " + str(s2v.similarity('run|VERB', 'think|VERB'))) + "\n"
results.write(str(write_string))

results.write("Most similar" + "\n")
write_string = ""
write_string += ("Most similar 'run' " + str(sense_vec_query("run|VERB"))) + "\n"
write_string += ("Most similar 'walked' " + str(sense_vec_query("walked|VERB"))) + "\n"
write_string += ("Most similar 'jogged' " + str(sense_vec_query("jogged|VERB"))) + "\n"
write_string += ("Most similar 'talked' " + str(sense_vec_query("talked|VERB"))) + "\n"
write_string += ("Most similar 'slept' " + str(sense_vec_query("slept|VERB"))) + "\n"
results.write(str(write_string))

results.write("Logical entailment" + "\n")
results.write("Compositions" + "\n")

results.write("Opposites" + "\n")

########################################################################################################################
