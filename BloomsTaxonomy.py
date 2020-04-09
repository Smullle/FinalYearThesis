import gensim
from sense2vec import Sense2Vec
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

w2v_model = "/home/shanesmullen/train/vmshare/models/word2vec/word2vec.model"
s2v_model = "/home/shanesmullen/train/vmshare/models/sense2vec"
r_data = "/home/shanesmullen/train/vmshare/results/bloom_r.txt"

knowledge = ["List", "Name", "Identify", "Reproduce"]
understanding = ["Describe", "Explain", "Classify", "Discuss"]
application = ["Apply", "Choose", "Employ", "Operate", "Practice"]
analysis = ["Compare", "Contrast", "Calculate", "Test", "Analyze"]
evaluation = ["Argue", "Assess", "Defend", "Judge", "Summarise"]
create = ["Construct", "Compose", "Create", "Design", "Propose"]

categories = [knowledge, understanding, application, analysis, evaluation, create]

w2v = gensim.models.Word2Vec.load(w2v_model)
print("Word2Vec Model Loaded")
s2v = Sense2Vec().from_disk(s2v_model)
print("Sense2Vec Model Loaded")

results = open(r_data, "w")
print("Results File Created at:", r_data)

results.write("Sense2Vec\n")
for cat in categories:
    for word in cat:
        comp_words = [w for w in cat if w != word]
        for comp in comp_words:
            results.write(str(s2v.similarity([word.lower() + '|VERB'], [comp.lower() + '|VERB'])) + ',')
    results.write("\n")

results.write("Word2Vec\n")
for cat in categories:
    for word in cat:
        comp_words = [w for w in cat if w != word]
        for comp in comp_words:
            results.write(str(w2v.similarity(word.lower(), comp.lower())) + ',')
    results.write("\n")
