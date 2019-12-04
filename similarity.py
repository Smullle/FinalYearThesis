import gensim
from sense2vec import Sense2Vec

# new push needs DocBin from spacy
# recent push not updated in pip

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True)

print("Google Vectors")

print("2similarity('run', 'tell') ", model.similarity('run', 'tell'))

print("2model2.similarity('run', 'walk')", model.similarity('run', 'walk'))

print("2model2.similarity('run', 'drive')", model.similarity('run', 'drive'))

print("2model2.similarity('run', 'talk')", model.similarity('run', 'talk'))

print("2model2.similarity('run', 'hatter')", model.similarity('run', 'hatter'))

print("2model2.similarity('run', 'think')", model.similarity('run', 'think'))

print("Wiki Vectors")

model = gensim.models.Word2Vec.load('word2vec.model')

print("2similarity('run', 'tell') ", model.similarity('run', 'tell'))

print("2model2.similarity('run', 'walk')", model.similarity('run', 'walk'))

print("2model2.similarity('run', 'drive')", model.similarity('run', 'drive'))

print("2model2.similarity('run', 'talk')", model.similarity('run', 'talk'))

print("2model2.similarity('run', 'hatter')", model.similarity('run', 'hatter'))

print("2model2.similarity('run', 'think')", model.similarity('run', 'think'))
