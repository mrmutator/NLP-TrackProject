import gzip
from nltk.tokenize import RegexpTokenizer
import glob
import sys


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, file_name, max_limit=None):
        self.file = gzip.open(file_name, "rb")
        self.max_limit = max_limit
        self.tokenizer = RegexpTokenizer(r'\w+')


    def __iter__(self):
        """
        Generator that returns a list of tokens for each sentence.
        :return: list of tokens
        """
        self.file.seek(0)
        c = 0
        for line in self.file:
            c += 1
            if self.max_limit and c > self.max_limit:
                break
            yield self.tokenizer.tokenize(line.decode("utf-8"))

    def get_raw_corpus(self):
        """ Generator that returns one sentence each time as a string (untokenized)
        :return:one sentence (string)
        """
        self.file.seek(0)
        c = 0
        for line in self.file:
            c += 1
            if self.max_limit and c > self.max_limit:
                break
            yield line.decode("utf-8")


if __name__ == "__main__":
    corpus = CorpusReader("data/europarl-v7.de.gz", max_limit=100)

    for tokens in corpus:
        print tokens