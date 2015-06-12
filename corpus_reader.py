import gzip
from nltk.tokenize import RegexpTokenizer
import glob
import sys


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, files, max_limit=None):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.max_limit = max_limit

    def __iter__(self):
        """
        Generator that returns a list of tokens for each sentence.
        :return: list of tokens
        """
        for f in self.files:
            print "Processing ", f
            c = 0
            for line in gzip.open(f, "rb"):
                c += 1
                if c > self.max_limit:
                    break
                yield self.tokenizer.tokenize(line)