# Sense2Vec: Investigating the Advantage of Introducing Part of Speech Tagging When Training Word Embeddings

A final year project investigating two methods for training word embeddings; Word2Vec and 
[Sense2Vec](https://github.com/explosion/sense2vec). Included are the 3 files used to produce the values used within 
Chapter 4 results; [BloomsTanonomy.py](../../BloomsTaxonomy.py), 
[ResultsCsv.py](../../ResultsCsv.py) and 
[RanWordsTxt.py](../../RanWordsTxt.py).

To produce the boxplots in sections 4.6 and 4.7 the following R files were used; [WordnetNoun.r](../../WordnetNoun.r),
[WordnetVerb.r](../../WordnetVerb.r) and [bloom.r](../../bloom.r)

## Abstract
Many methods for producing word embeddings have been developed but one project considered the standard, Word2Vec is 
able to extract context and give the illusion of knowledge when trained on a corpus of text. However it is not without 
its pitfalls, when investigating Word2Vec's ability to group verbs it does not produce the same level of quality as with
 noun embeddings. The Sense2Vec project introduces part of speech tagging to the training process allowing for a more 
 refined level of context to be obtained. When investigating verb embeddings specifically a greater relationship was 
 found between verb tenses than originally produced by Word2Vec. Also by making a distinction between words that may be 
 considered a noun or a verb depending on context, a more refined list of similar embeddings is produced.

## Installation of Required Libraries

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install requirements.txt
```

## Training

Download and store a large corpus of Wikipedia text located [here](https://dumps.wikimedia.org/enwiki/) it is 
recommended to used a corpus which contains at least 1 billion words.

## WordLists
The two words lists; [nounlist.txt](../../WordLists/nounlist.txt) and [verblist.txt](../../WordLists/verblist.txt) 
required for [RanWordsTxt.py](../../RanWordsTxt.py).

## LaTeX
Contains the set of .tex files to construct the pdf and a [.bib](../../LaTeX/Thesis.bib) file containing information for
 referencing. [Images](../../LaTeX/images) and [figures](../../LaTeX/images) used are also located within this 
directory.

## Acknowledgments
I’d like to extend my gratitude to my supervisor Dr Diarmuid O'Donoghue, for first introducing the idea and providing 
help and guidance whenever needed. I am grateful for the work done by the Sense2Vec team and the provided documentation 
for their project. Also of course all friends and family who provided helpful input and suggestions throughout the 
project.