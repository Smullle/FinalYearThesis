# Python program to generate word vectors using Word2Vec


# importing all necessary modules

from nltk.tokenize import sent_tokenize, word_tokenize

import warnings

warnings.filterwarnings(action='ignore')

import gensim

from gensim.models import Word2Vec

#  Reads ‘alice.txt’ file

sample = open("C:\\Users\\user\\Desktop\\alice.txt", "r")

s = sample.read()

# Replaces escape character with space

f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file

for i in sent_tokenize(f):

    temp = []

    # tokenize the sentence into words

    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

######################################

############## StopWords #############

######################################

# Load library

from nltk.corpus import stopwords

# You will have to download the set of stop words the first time

import nltk

# nltk.download('stopwords')


stop_words = stopwords.words('english')

# Show stop words

# print(data[0])

# new_words = [word for word in words if word not in stopwords]

[word for word in data if word not in stop_words]

for sent in data:

    for word in sent:

        if word in stop_words:
            sent.remove(word)

# print(data[0])


# stop()

######################################

############### Word2Vec #############

######################################

print("Word2Vec")

# Create CBOW model

model1 = gensim.models.Word2Vec(data, min_count=1,

                                size=100, window=3)

# Print results

print("Cosine similarity between 'alice' " +

      "and 'wonderland' - CBOW : ",

      model1.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +

      "and 'machines' - CBOW : ",

      model1.similarity('alice', 'machines'))

print("1similarity('run', 'tell') ", model1.similarity('run', 'tell'))

print("model1.similarity('run', 'walk')", model1.similarity('run', 'walk'))

print("model1.similarity('run', 'drive')", model1.similarity('run', 'drive'))

print("model1.similarity('run', 'talk')", model1.similarity('run', 'talk'))

print("model1.similarity('run', 'see')", model1.similarity('run', 'see'))

print("model1.similarity('get, 'eat')", model1.similarity('get', 'eat'))

# Create Skip Gram model

model2 = gensim.models.Word2Vec(data, min_count=1, size=100,

                                window=10, sg=1)

# Print results

print("Cosine similarity between 'alice' " +

      "and 'wonderland' - Skip Gram : ",

      model2.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +

      "and 'machines' - Skip Gram : ",

      model2.similarity('alice', 'machines'))

print("2similarity('run', 'tell') ", model2.similarity('run', 'tell'))

print("2model2.similarity('run', 'walk')", model2.similarity('run', 'walk'))

print("2model2.similarity('run', 'drive')", model2.similarity('run', 'drive'))

print("2model2.similarity('run', 'talk')", model2.similarity('run', 'talk'))

print("2model2.similarity('run', 'hatter')", model2.similarity('run', 'hatter'))

print("2model2.similarity('run', 'think')", model2.similarity('run', 'think'))

# print(model1['run'])