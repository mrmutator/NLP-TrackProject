import gzip
from nltk.tokenize import RegexpTokenizer
import glob
import sys


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, files):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __iter__(self):
        """
        Generator that returns a list of tokens for each sentence.
        :return: list of tokens
        """
        for f in self.files:
            print "Processing ", f
            for line in gzip.open(f, "rb"):
                yield self.tokenizer.tokenize(line)