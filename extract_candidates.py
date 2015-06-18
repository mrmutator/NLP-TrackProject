__author__ = 'rwechsler'
import sys
import time
import datetime
import gensim
import dawg
from collections import defaultdict
import cPickle as pickle
import numpy as np
from annoy import AnnoyIndex
import argparse


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def build_vocabulary(word2vec_model, min_length=0):
    """
    Build prefix and suffix vocabulary.
    :param corpus: CorpusReader object
    :param min_length: minimum character length of words to be considered
    :return: prefix_vocabulary and suffix_vocabulary
    """

    prefix_vocab = set()
    suffix_vocab = set()
    for tok in word2vec_model.vocab:
            if len(tok) >= min_length:
                prefix_vocab.add(tok)
                suffix_vocab.add(tok[::-1])

    return prefix_vocab, suffix_vocab

def add_prefix_combinations(combinations, prefix_vocab, dawg_model, fugenlaute=[]):
    fugenlaute += [""]
    for word in prefix_vocab:
        for prefix in dawg_model.prefixes(word)[:-1]: # last word is the word itself
            rest = word[len(prefix):]
            # Consider fugenlaute
            for fl in fugenlaute:
                if rest.startswith(fl):
                    if rest[len(fl):].title() in dawg_model:
                        combinations[prefix].add((fl, rest[len(fl):].title()))
                        # break
                    elif rest[len(fl):] in dawg_model:
                        combinations[prefix].add((fl, rest[len(fl):]))
                        # break

    return

def add_suffix_combinations(combinations, suffix_vocab, lower_suffix_dawg_model, fugenlaute=[""]):
    fugenlaute = [fl[::-1] for fl in fugenlaute] + [""]
    for word in suffix_vocab:
        for suffix in lower_suffix_dawg_model.prefixes(word): # last word is the word itself
            rest = word[len(suffix):]
            # Consider fugenlaute
            for fl in fugenlaute:
                if rest.startswith(fl):
                    if rest[len(fl):] in suffix_vocab:
                        upper_suffix = suffix[::-1].title()[::-1]
                        if upper_suffix in suffix_vocab:
                            combinations[rest[len(fl):][::-1]].add((fl[::-1], upper_suffix[::-1]))
                        elif suffix in suffix_vocab:
                            combinations[rest[len(fl):][::-1]].add((fl[::-1], suffix[::-1]))

                        # break

    return

def build_annoy_tree(word2vec_model,  output_file_name, n_trees=100):
    tree = AnnoyIndex(word2vec_model.layer1_size)
    for i, word in enumerate(word2vec_model.index2word):
        tree.add_item(i, list(word2vec_model[word]))

    tree.build(n_trees)


    tree.save(output_file_name)

    return output_file_name


def produce_candidate_index(candidates, word2vec_model, output_file_name):
    w2vv = word2vec_model.vocab
    candidate_index = dict()
    for prefix in candidates:
        candidate_index[prefix] = set()
        for tpl in candidates[prefix]:
            compound = prefix + tpl[0] + tpl[1].lower()
            tail = tpl[1]
            try:
                candidate_index[prefix].add((w2vv[compound].index, w2vv[tail].index))
            except KeyError:
                pass
                # here is just weird stuff: "GeorgeMichael", "VisitBerlin", ...
                # e.g. AnstragsstellerInnen
                # compound = prefix + tpl[0] + tpl[1]
                # tail = tpl[1]
                # candidate_index[prefix].add((w2vv[compound].index, w2vv[tail].index))


    pickle.dump(candidate_index, open(output_file_name, "wb"))



if __name__ == "__main__":

    ###------------Parameters-----------------###
    min_word_length = 5
    fugenlaute = ["es", "s"]
    n_annoy_trees = 100
    ###---------------------------------------###




    parser = argparse.ArgumentParser(description='Extracte candidates')

    parser.add_argument('-w', action="store", dest="word2vec_file", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', action="store", dest="dawg_name", help="file_prefix for two files <..>.prefixes and <..>.suffixes")
    group.add_argument('-b', action="store", dest="build_dawg_name")
    parser.add_argument('-c', action="store", dest="output_candidate_file", required=True)
    parser.add_argument('-o', action="store", dest="output_annoy_file")
    parser.add_argument('-i', action="store", dest="candidate_index_file", required=True)
    parser.add_argument('-l', action="store", type=int, dest="min_word_length", default=min_word_length)
    parser.add_argument('-n', action="store", dest="n_annoy_trees", type=int, default=n_annoy_trees)

    arguments = parser.parse_args(sys.argv[1:])



    print timestamp(), "loading word2ved model"
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(arguments.word2vec_file, binary=True)

    print timestamp(), "building vocabulary ..."
    prefix_vocab, suffix_vocab = build_vocabulary(word2vec_model, min_length=arguments.min_word_length)

    if arguments.build_dawg_name:
        print timestamp(), "building dawg models"
        dawg_model = dawg.DAWG(prefix_vocab)
        lower_suffix_dawg_model = dawg.DAWG(set(w.lower() for w in suffix_vocab))
        print timestamp(), "saving dawg models"
        dawg_model.save(arguments.build_dawg_name + ".prefixes")
        lower_suffix_dawg_model.save(arguments.build_dawg_name + ".suffixes")
    else:
        print timestamp(), "loading dawg models"
        dawg_model = dawg.DAWG()
        dawg_model.load(arguments.dawg_name + ".prefixes")
        lower_suffix_dawg_model = dawg.DAWG(arguments.dawg_name + ".suffixes")


    candidates = defaultdict(set)
    print timestamp(), "prefix pass ..."
    add_prefix_combinations(candidates, prefix_vocab, dawg_model, fugenlaute=fugenlaute)
    print timestamp(), "suffix pass ..."
    add_suffix_combinations(candidates, suffix_vocab, lower_suffix_dawg_model, fugenlaute=fugenlaute)

    print timestamp(), "pickling model ..."
    pickle.dump(candidates, open(arguments.output_candidate_file, "wb"))

    if arguments.output_annoy_file:
        print timestamp(), "building annoy tree"
        annoy_tree_file = build_annoy_tree(word2vec_model, n_trees=arguments.n_annoy_trees, output_file_name=arguments.output_annoy_file)

    print timestamp(), "producing candidate index file"
    produce_candidate_index(candidates, word2vec_model, arguments.candidate_index_file)

    print timestamp(), "done."


    # some statistics
    print "Vocabulary size: ", len(prefix_vocab)
    for k, v in candidates.items()[:40]:
        try:
            print k.encode("utf-8"), v
        except:
            pass


    tuples = sorted([(k, len(v)) for k,v in candidates.items()], key= lambda tup: tup[1], reverse=True)
    print "------"
    print "Longest keys: "
    for k, v in tuples[:40]:
        try:
            print k.encode("utf-8"), v
        except:
            pass

    keys, lengths = zip(*tuples)

    print "----"
    print "Keys: ", len(candidates)
    print "Longest key: ", keys[np.argmax(lengths)], np.max(lengths)
    print "Average length: ", np.mean(lengths)


