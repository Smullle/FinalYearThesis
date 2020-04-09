"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py
"""

from gensim.corpora import WikiCorpus
import warnings

warnings.filterwarnings(action='ignore')


def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""

    output = open(out_f, 'w', encoding='utf-8')
    wiki = WikiCorpus(in_f)

    i = 0
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i = i + 1
        if i % 10000 == 0:
            print('Processed ' + str(i) + ' articles')
    output.close()
    print('Processing complete!')


if __name__ == '__main__':
    in_f = "E:\enwiki-latest-pages-articles-multistream.xml.bz2"
    out_f = "E:\wiki.txt"
    make_corpus(in_f, out_f)
