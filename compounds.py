__author__ = 'rwechsler'

from CorpusReader import CorpusReader
from _collections import defaultdict

def build_vocabulary(corpus, min_length=0):

    vocab = defaultdict(int)
    for tokens in corpus:
        for tok in tokens:
            if len(tok) >= min_length:
                vocab[tok] += 1

    return vocab


def get_combinations(vocab):
    compound_dict = defaultdict(int)
    combinations = defaultdict(lambda: defaultdict(int))
    for w in vocab:
        if w[0].upper() == w[0]:
            for head in vocab:
                head= head.lower()
                combo = vocab.get(w + head, 0)
                if combo:
                    combinations[head][w] += 1
                    compound_dict[w + "-" + head] += 1

    return combinations, compound_dict




if __name__ == "__main__":

    corpus = CorpusReader("data/europarl-v7.de.gz", max_limit=10000)
    print "building vocab"
    vocab = build_vocabulary(corpus, min_length=4)
    print len(vocab)
    print "building combinations"
    combinations, compound_dict = get_combinations(vocab)
    print "print combinations built"
    # print most frequent combinations:

    for compound in sorted(compound_dict, key=compound_dict.get, reverse=True)[:50]:
        print compound, compound_dict[compound]
