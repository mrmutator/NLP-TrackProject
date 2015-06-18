__author__ = 'rwechsler'
import cPickle as pickle
import gensim
import itertools
import random
from annoy import AnnoyIndex
import multiprocessing as mp
import sys
import argparse
import time
import datetime
import numpy as np


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)


def load_annoy_tree(model_file_name, vector_dims):
    tree = AnnoyIndex(vector_dims)
    tree.load(model_file_name)
    return tree

def annoy_knn(annoy_tree, vector, true_index, k=100):
    neighbours = annoy_tree.get_nns_by_vector(list(vector), k)
    if true_index in neighbours:
        return True
    else:
        return False


def evaluate_set(prefix, tails, annoy_tree_file, vector_dims, rank_threshold=100, sample_size=1000):
    counts = dict()
    counts[True] = 0
    counts[False] = 0
    annoy_tree = load_annoy_tree(annoy_tree_file, vector_dims)
    if len(tails) > sample_size:
        tails = random.sample(tails, sample_size)
    for (comp1, tail1), (comp2, tail2) in itertools.combinations(tails, 2):
        try:
            diff = np.array(annoy_tree.get_item_vector(comp2))- np.array(annoy_tree.get_item_vector(tail2))
            predicted = np.array(annoy_tree.get_item_vector(tail1)) + diff

            result = annoy_knn(annoy_tree, predicted, comp1, rank_threshold)

            counts[result] += 1

        except KeyError:
            pass

    annoy_tree.unload(annoy_tree_file)

    return prefix, float(counts[True]) / (counts[True] + counts[False]) if counts[True] + counts[False] > 0 else prefix, 0.0

def test_pair(pair1, pair2, word2vec_model, k=100, show=30):
    """
    Only used in interactive mode so far.
    :param pair1:
    :param pair2:
    :param word2vec_model:
    :param k:
    :param show:
    :return:
    """
    prefix = pair1[0]
    fl1 = pair1[1]
    tail1 = pair1[2]
    prefix2 = pair2[0]
    fl2 = pair2[1]
    tail2 = pair2[2]
    assert prefix == prefix2

    diff = word2vec_model[prefix + fl2 + tail2.lower()] - word2vec_model[tail2]
    predicted = word2vec_model[tail1] + diff

    true_word = prefix + fl1 + tail1.lower()

    neighbours = word2vec_model.most_similar([predicted], topn=k)

    print neighbours[:show]
    neighbours, _ = zip(*neighbours)
    print "Found: ", true_word in neighbours


def candidate_generator(candidates, annoy_tree_file, vector_dims, rank_threshold, sample_size):
    for prefix in candidates:
        yield (prefix, candidates[prefix], annoy_tree_file, vector_dims, rank_threshold, sample_size)

def mp_wrapper_evaluate_set(argument):
    return evaluate_set(*argument)


def evaluate_candidates(candidates, annoy_tree_file, vector_dims, rank_threshold=100, sample_size=500, processes=4):
    pool = mp.Pool(processes=processes)

    arguments = candidate_generator(candidates, annoy_tree_file, vector_dims, rank_threshold, sample_size)
    results = pool.map(mp_wrapper_evaluate_set, arguments)

    return results


if __name__ == "__main__":

    #### Default Parameters-------------------------------------------####
    rank_threshold = 100
    sample_set_size = 500
    n_processes = 2
    ####End-Parametes-------------------------------------------------####


    parser = argparse.ArgumentParser(description='Evaluate candidates')
    parser.add_argument('-d', action="store", dest="vector_dims", type=int, required=True)
    parser.add_argument('-t', action="store", dest="annoy_tree_file", required=True)
    parser.add_argument('-c', action="store", dest="candidates_index_file", required=True)
    parser.add_argument('-o', action="store", dest="result_output_file", required=True)
    parser.add_argument('-p', action="store", dest="n_processes", type=int, default=n_processes)
    parser.add_argument('-s', action="store", dest="sample_set_size", type=int, default=sample_set_size)
    parser.add_argument('-r', action="store", dest="rank_threshold", type=int, default=rank_threshold)

    arguments = parser.parse_args(sys.argv[1:])


    print timestamp(), "loading candidates"
    candidates = load_candidate_dump(arguments.candidates_index_file)

    print timestamp(), "evaluating candidates"
    results = evaluate_candidates(candidates, arguments.annoy_tree_file, arguments.vector_dims, rank_threshold=arguments.rank_threshold,
                                  sample_size=arguments.sample_set_size, processes=arguments.n_processes)

    print timestamp(), "pickling"
    pickle.dump(results, open(arguments.result_output_file, "wb"))

    print timestamp(), "done"
