# Sense2Vec: Investigating the Advantage of Introducing Part of Speech Tagging When Training Word Embeddings

A final year project investigating two methods for training word embeddings; Word2Vec and 
[Sense2Vec](https://github.com/explosion/sense2vec). Included are the 3 files used to produce the values used within 
Chapter 4 results; [BloomsTanonomy.py](../../BloomsTaxonomy.py), 
[ResultsCsv.py](../../ResultsCsv.py) and 
[RanWordsTxt.py](../../RanWordsTxt.py).

To produce the boxplots in sections 4.6 and 4.7 the following R files were used; [WordnetNoun.r](../../WordnetNoun.r),
[WordnetVerb.r](../../WordnetVerb.r) and [bloom.r](../../bloom.r)

## Installation of Required Libraries

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install requirements.txt
```

## Training

Download and store a large corpus of Wikipedia text located [here](https://dumps.wikimedia.org/enwiki/) it is 
recommended to used a corpus which contains at least 1 billion words.

## Acknowledgments
I’d like to extend my gratitude to my supervisor Dr Diarmuid O'Donoghue, for first introducing the idea and providing 
help and guidance whenever needed. I am grateful for the work done by the Sense2Vec team and the provided documentation 
for their project. Also of course all friends and family who provided helpful input and suggestions throughout the 
project.