import stanfordnlp
from gensim.test.utils import datapath
from gensim import utils
import plac


@plac.annotations(
    in_dir=("Directory with preprocessed .s2v files", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
)
def main(
    in_dir, out_dir,
):
    """
    Step 1: parse and part of speech (POS) tag each word in the corpus

    This process is completed using the stanford parser and uses the following pipelines:
    - TokenizeProcessor
    - MWTProcessor
    - POSProcessor
    No lemmatization is preformed to save processing time with large corpus files, more info found here:
    https://stanfordnlp.github.io/stanfordnlp/index.html

    Expects a corpus wile within in_dir in txt format
    Outputs a tagged txt file to the supplied output directory with out_dir
    """

    class MyCorpus(object):
        """An iterator that yields sentences (lists of str)."""

        def __iter__(self):
            corpus_path = datapath(in_dir)
            for line in open(corpus_path, errors="ignore"):
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)

    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')  # This sets up a default neural pipeline in English

    total_words = 0

    data = MyCorpus()
    output = open(out_dir, "a")
    for line in data:
        total_words += len(line)
        try:
            if len(line) < 3000:
                doc = nlp(' '.join(line))
                for sent in doc.sentences:
                    for word in sent.words:
                        output.write(*[word.text + "|" + word.upos + " "])
            else:
                segments = ([line[x:x + 3000] for x in range(0, len(line), 3000)])
                for segment in segments:
                    doc = nlp(' '.join(segment))
                    # doc.sentences[0].print_dependencies()
                    for sent in doc.sentences:
                        for word in sent.words:
                            output.write(*[word.text + "|" + word.upos + " "])
        except AssertionError:
            print(len(line))

        print(total_words)

