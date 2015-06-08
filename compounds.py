__author__ = 'rwechsler'

from CorpusReader import CorpusReader
from _collections import defaultdict
import multiprocessing as mp
import time
import cPickle as pickle


def build_vocabulary(corpus, min_length=0):

    vocab = defaultdict(int)
    for tokens in corpus:
        for tok in tokens:
            if len(tok) >= min_length:
                vocab[tok] += 1

    return vocab


def get_combinations(vocab, threads=4):

    size = int(round(len(vocab) / float(threads)))
    splits = [(size*i, (i+1) * size) for i in range(threads)]
    splits[-1] = (splits[-1][0], None) # last part can be longer if more leftovers

    heads = vocab.keys()

    pool = mp.Pool(processes=threads)

    results = [pool.apply_async(try_combinations, args=(heads[i:j], vocab)) for i, j in splits]

    output = [p.get() for p in results]


    combinations = dict()

    for comb_dict in output:
        combinations.update(comb_dict)

    return combinations



def try_combinations(heads, vocab):
    combinations = defaultdict(set)
    for w in vocab:
        for head in heads:
            head= head.lower()
            combo = vocab.get(w + head, 0)
            if combo:
                combinations[head].add(w)

    return combinations


if __name__ == "__main__":

    start = time.time()

    corpus = CorpusReader("data/europarl-v7.de.gz", max_limit=1000)
    print "building vocab"
    vocab = build_vocabulary(corpus, min_length=4)
    print len(vocab)
    print "building combinations"
    combinations = get_combinations(vocab, threads=8)
    print "combinations built"
    # print most frequent combinations:

    end = time.time()

    print "pickling"

    pickle.dump(combinations, open("output.p", "wb"))

    print combinations

    print "Time: ", end-start