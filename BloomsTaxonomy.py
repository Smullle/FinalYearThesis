import gensim
from sense2vec import Sense2Vec


w2v_model = "E:\\WikiData\\Trained\\word2vec\\5GB\\word2vec.model"
s2v_model = "E:\\WikiData\\Trained\\sense2vec\\RedditVecs\\sense2vec-vectors"
r_data = "E:\\Results\\bloom_r.txt"

knowledge = ["List", "Name", "Identify", "Reproduce"]
understanding = ["Describe", "Explain", "Classify", "Discuss"]
application = ["Apply", "Choose", "Employ", "Operate", "Practice"]
analysis = ["Compare", "Contrast", "Calculate", "Test", "Analyze"]
evaluation = ["Argue", "Assess", "Defend", "Judge", "Summarise"]
create = ["Construct", "Compose", "Create", "Design", "Propose"]

categories = [knowledge, understanding, application, analysis, evaluation, create]

results = open(r_data, "w")

s2v = Sense2Vec().from_disk(s2v_model)

results.write("Sense2Vec\n")
for cat in categories:
    for word in cat:
        comp_words = [w for w in cat if w != word]
        for comp in comp_words:
            results.write(str(s2v.similarity([word.lower() + '|VERB'], [comp.lower() + '|VERB'])) + ',')
    results.write("\n")

w2v = gensim.models.Word2Vec.load(w2v_model)

results.write("Word2Vec\n")
for cat in categories:
    for word in cat:
        comp_words = [w for w in cat if w != word]
        for comp in comp_words:
            results.write(str(w2v.similarity(word.lower(), comp.lower())) + ',')
    results.write("\n")
