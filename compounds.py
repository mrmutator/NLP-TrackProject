__author__ = 'rwechsler'

from CorpusReader import CorpusReader
from _collections import defaultdict
import multiprocessing as mp
import time
import cPickle as pickle

def dd():
    return defaultdict(int)


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


    # processes = [mp.Process(target=try_combinations, args=(heads[i:j], vocab, output)) for i, j in splits]
    #
    # for p in processes:
    #     p.start()
    #
    # print "running"
    #
    # for p in processes:
    #     p.join()
    #
    # print "done"

    pool = mp.Pool(processes=threads)

    results = [pool.apply_async(try_combinations, args=(heads[i:j], vocab)) for i, j in splits]

    output = [p.get() for p in results]


    combinations = dict()
    compound_dict = dict()

    for i, tup in enumerate(output):
        combinations.update(tup[0])
        compound_dict.update(tup[1])

    return combinations, compound_dict



def try_combinations(heads, vocab):
    combinations = defaultdict(dd)
    compound_dict = defaultdict(int)
    for w in vocab:
        if w[0].upper() == w[0]:
            for head in heads:
                head= head.lower()
                combo = vocab.get(w + head, 0)
                if combo:
                    combinations[head][w] += 1
                    compound_dict[w + "-" + head] += 1

    return combinations, compound_dict


if __name__ == "__main__":

    start = time.time()

    corpus = CorpusReader("data/europarl-v7.de.gz", max_limit=200)
    print "building vocab"
    vocab = build_vocabulary(corpus, min_length=4)
    print len(vocab)
    print "building combinations"
    combinations, compound_dict = get_combinations(vocab, threads=5)
    print "combinations built"
    # print most frequent combinations:

    end = time.time()

    print "pickling"

    pickle.dump([combinations, compound_dict], open("output.p", "wb"))


    for compound in sorted(compound_dict, key=compound_dict.get, reverse=True)[:50]:
        print compound, compound_dict[compound]

    print "Time: ", end-start