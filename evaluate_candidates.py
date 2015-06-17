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


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

def build_annoy_tree(word2vec_model, n_trees=100, output_file_name=None):
    tree = AnnoyIndex(word2vec_model.layer1_size)
    for i, word in enumerate(word2vec_model.index2word):
        tree.add_item(i, list(word2vec_model[word]))

    tree.build(n_trees) #

    if output_file_name:
        tree.save(output_file_name)

    return tree

def load_annoy_tree(model_file_name, word2vec_model):
    tree = AnnoyIndex(word2vec_model.layer1_size)
    tree.load(model_file_name)
    return tree

def annoy_knn(annoy_tree, vector, true_index, k=100):
    neighbours = annoy_tree.get_nns_by_vector(list(vector), k)
    if true_index in neighbours:
        return True
    else:
        return False

def word2vec_knn(word2vec_model, vector, true_word, k=100):
    neighbours, _ = zip(*word2vec_model.most_similar([vector], topn=k))
    if true_word in neighbours:
        return True
    else:
        return False


def evaluate_set(prefix, tails, word2vec_model, annoy_tree, rank_threshold=100, sample_size=1000):
    counts = dict()
    counts[True] = 0
    counts[False] = 0
    if len(tails) > sample_size:
        tails = random.sample(tails, sample_size)
    for (fl1, tail1), (fl2, tail2) in itertools.combinations(tails, 2):
        try:
            diff = word2vec_model[prefix + fl2 + tail2.lower()] - word2vec_model[tail2]
            predicted = word2vec_model[tail1] + diff
            true_word = prefix + fl1 + tail1.lower()
            true_index = word2vec_model.vocab[true_word].index

            result = annoy_knn(annoy_tree, predicted, true_index, rank_threshold)
            #result = word2vec_knn(word2vec_model, predicted, true_word, rank_threshold)

            counts[result] += 1

        except KeyError:
            pass

    return float(counts[True]) / (counts[True] + counts[False]) if counts[True] + counts[False] > 0 else 0

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
    # cosine similarity
    # true_vector = word2vec_model[prefix + fl1 + tail1]
    # similarities.append(np.dot(gensim.matutils.unitvec(true_vector), gensim.matutils.unitvec(predicted)))

    true_word = prefix + fl1 + tail1.lower()

    neighbours = word2vec_model.most_similar([predicted], topn=k)

    print neighbours[:show]
    neighbours, _ = zip(*neighbours)
    print "Found: ", true_word in neighbours


def candidate_generator(candidates, annoy_tree, word2vec_model, rank_threshold, sample_size):
    for prefix in candidates:
        yield (prefix, candidates[prefix], word2vec_model, annoy_tree, rank_threshold, sample_size)


def evaluate_candidates(candidates, annoy_tree, word2vec_model, rank_threshold=100, sample_size=500, processes=4):
    pool = mp.Pool(processes=processes)

    arguments = candidate_generator(candidates, annoy_tree, word2vec_model, rank_threshold, sample_size)
    results = pool.map(evaluate_set, arguments)

    return zip(candidates.keys(), results)


if __name__ == "__main__":

    #### Parameters-------------------------------------------####
    rank_threshold = 100
    sample_set_size = 500
    n_annoy_trees = 10
    n_processes = 2
    ####End-Parametes-----------------------------------------####


    parser = argparse.ArgumentParser(description='Evaluate candidates')

    parser.add_argument('-w', action="store", dest="word2vec_file", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', action="store", dest="annoy_tree_file")
    group.add_argument('-b', action="store", dest="output_tree_file")
    parser.add_argument('-c', action="store", dest="candidates_file")
    parser.add_argument('-o', action="store", dest="result_output_file")
    parser.add_argument('-p', action="store", dest="n_processes", type=int, default=n_processes)

    arguments = parser.parse_args(sys.argv[1:])

    print timestamp(), "loading word2vec model"
    word2vec_model = load_word2vecmodel(arguments.word2vec_file)

    if not arguments.annoy_tree_file:

        print timestamp(), "building annoy tree"
        annoy_tree = build_annoy_tree(word2vec_model, n_trees=n_annoy_trees, output_file_name=arguments.output_tree_file)

    else:
        print timestamp(), "loading annoy tree"
        annoy_tree = load_annoy_tree(arguments.annoy_tree_file, word2vec_model)

    if arguments.candidates_file and arguments.result_output_file:
        print timestamp(), "loading candidates"
        candidates = load_candidate_dump(arguments.candidates_file)

        print timestamp(), "Evaluating candidates"
        results = evaluate_candidates(candidates, annoy_tree, word2vec_model, rank_threshold=rank_threshold,
                                      sample_size=sample_set_size, processes=n_processes)

        print timestamp(), "pickling"
        pickle.dump(results, open(arguments.result_output_file, "wb"))


    print timestamp(), "done"



