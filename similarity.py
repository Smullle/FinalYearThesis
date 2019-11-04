import gensim
import sense2vec


# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True)

print("2similarity('run', 'tell') ", model.similarity('run', 'tell'))

print("2model2.similarity('run', 'walk')", model.similarity('run', 'walk'))

print("2model2.similarity('run', 'drive')", model.similarity('run', 'drive'))

print("2model2.similarity('run', 'talk')", model.similarity('run', 'talk'))

print("2model2.similarity('run', 'hatter')", model.similarity('run', 'hatter'))

print("2model2.similarity('run', 'think')", model.similarity('run', 'think'))