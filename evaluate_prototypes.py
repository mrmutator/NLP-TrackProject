__author__ = 'rwechsler'
import datetime
import time
import cPickle as pickle
from annoy import AnnoyIndex
import gensim
import argparse
import numpy as np
import sys
import random
from scipy import spatial
import multiprocessing as mp
from collections import defaultdict
import codecs

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_annoy_tree(model_file_name, vector_dims):
    tree = AnnoyIndex(vector_dims)
    tree.load(model_file_name)
    return tree

def load_prototype_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

def get_rank_annoy_knn(annoy_tree, vector, true_index, k=100):
    neighbours = annoy_tree.get_nns_by_vector(list(vector), k)
    try:
        return neighbours.index(true_index) + 1
    except ValueError:
        return 0

def candidate_generator(evaluation_set, rank_threshold, sim_threshold):
    for prefix_prototype_pair in evaluation_set:
        yield (prefix_prototype_pair, evaluation_set[prefix_prototype_pair], rank_threshold, sim_threshold)

def mp_wrapper_evaluate_set(argument):
    return evaluate_set(*argument)

def get_nn_hitrate(ranks):
    return (len(ranks) - ranks.count(0)) / float(len(ranks))

def get_sim_hitrate(similarities, threshold):
    return np.sum([1 for s in similarities if s >= threshold]) / float(len(similarities))

def get_average_rank(ranks):
    return np.mean([r for r in ranks if r > 0] or 0)

def get_average_similarity(similarities):
    return np.mean(similarities)

def get_hitrate(ranks, similarities, threshold):
    count = 0
    for i, r in enumerate(ranks):
        if r > 0 and similarities[i] >= threshold:
            count += 1
    return count / float(len(ranks))

def get_word_representation(prefix, comp_index, tail_index, word2vec_model):
        comp = word2vec_model.index2word[comp_index]
        tail = word2vec_model.index2word[tail_index]
        fl = comp[len(prefix):-len(tail)]

        if fl:
            fl = "[" + fl + "]"

        return fl + tail


if __name__ == "__main__":

    #### Default Parameters-------------------------------------------####
    rank_threshold = 30
    sim_threshold = 0.5
    sample_set_size = np.inf
    n_processes = 2
    ####End-Parametes-------------------------------------------------####


    parser = argparse.ArgumentParser(description='Evaluate candidates')
    parser.add_argument('-w', action='store', dest="word2vec_file", required=True)
    parser.add_argument('-v', action="store", dest="prototypes_file", required=True)
    parser.add_argument('-d', action="store", dest="vector_dims", type=int, required=True)
    parser.add_argument('-t', action="store", dest="annoy_tree_file", required=True)
    parser.add_argument('-c', action="store", dest="candidates_index_file")
    parser.add_argument('-o', action="store", dest="result_output_file", required=True)
    parser.add_argument('-p', action="store", dest="n_processes", type=int, default=n_processes)
    parser.add_argument('-s', action="store", dest="sample_set_size", type=int, default=sample_set_size)
    parser.add_argument('-r', action="store", dest="rank_threshold", type=int, default=rank_threshold)
    parser.add_argument('-z', action="store", dest="sim_threshold", type=float, default=sim_threshold)

    arguments = parser.parse_args(sys.argv[1:])


    print timestamp(), "loading word2vec model"
    word2vec_model = load_word2vecmodel(arguments.word2vec_file)

    print timestamp(), "loading prototypes"
    prototypes = load_prototype_dump(arguments.prototypes_file)

    if arguments.candidates_index_file:
        print timestamp(), "loading candidates"
        candidates = load_candidate_dump(arguments.candidates_index_file)


    evaluation_set = dict()
    # keys are (prefix, prototype_pair)
    for prefix in prototypes:
        for prototype, evidence_set in prototypes[prefix]:
            if arguments.candidates_index_file:
                evaluation_set[(prefix, prototype)] = candidates[prefix]
            else:
                evaluation_set[(prefix, prototype)] = evidence_set


    print timestamp(), "preprocess candidates"
    # only store vectors that we need. And sample already.
    word2vec_vectors = dict()
    for prototype_tup in evaluation_set:
        if len(evaluation_set[prototype_tup]) > arguments.sample_set_size:
            evaluation_set[prototype_tup] = set(random.sample(evaluation_set[prototype_tup], arguments.sample_set_size))
        for (i,j) in evaluation_set[prototype_tup]:
            word2vec_vectors[i] = np.array(word2vec_model.syn0[i])
            word2vec_vectors[j] = np.array(word2vec_model.syn0[j])
        word2vec_vectors[prototype_tup[1][0]] = np.array(word2vec_model.syn0[prototype_tup[1][0]])
        word2vec_vectors[prototype_tup[1][1]] = np.array(word2vec_model.syn0[prototype_tup[1][1]])

    del word2vec_model

    print timestamp(), "number of vectors: ", len(word2vec_vectors)

    print timestamp(), "load annoy tree"
    # global annoy_tree
    annoy_tree = load_annoy_tree(arguments.annoy_tree_file, arguments.vector_dims)

    def evaluate_set(prefix_prototype_pair, evidence_set, rank_threshold=100, sim_threshold=0.5):
        global annoy_tree
        global word2vec_vectors

        ranks = []
        similarities = []

        prefix, vector_pair = prefix_prototype_pair
        diff = word2vec_vectors[vector_pair[0]]- word2vec_vectors[vector_pair[1]]


        for comp, tail in evidence_set:
            predicted = word2vec_vectors[tail] + diff
            true_vector = word2vec_vectors[comp]
            rank = get_rank_annoy_knn(annoy_tree, predicted, comp, rank_threshold)
            ranks.append(rank)
            sim = spatial.distance.cosine(predicted, true_vector)
            similarities.append(sim)

        # returns hitrate, hitrate_nn, hitrate_sim, average_rank_if_found, average_similarity_if_found
        results = get_hitrate(ranks, similarities, threshold=sim_threshold), get_nn_hitrate(ranks), get_sim_hitrate(similarities, threshold=sim_threshold), get_average_rank(ranks), get_average_similarity(similarities)

        return (prefix_prototype_pair,results)

    print timestamp(), "evaluating candidates"
    pool = mp.Pool(processes=arguments.n_processes)
    params = candidate_generator(evaluation_set, arguments.rank_threshold, arguments.sim_threshold)
    results = pool.map(mp_wrapper_evaluate_set, params)


    print timestamp(), "pickling"
    pickle.dump(results, open(arguments.result_output_file, "wb"))


    print timestamp(), "loading word2vec model"
    word2vec_model = load_word2vecmodel(arguments.word2vec_file)

    print timestamp(), "mapping indices to word"
    scores = defaultdict(dict)
    for ((prefix, vector), eval_scores) in results:
        vector_repr = get_word_representation(prefix, vector[0], vector[1], word2vec_model)
        scores[prefix][vector_repr] = eval_scores

    print timestamp(), "writing result file"
    outfile = codecs.open(arguments.result_output_file, "w", "utf-8")
    for prefix in scores:
        for vector in scores[prefix]:
            outfile.write("\t".join([prefix, vector] + map(str, scores[prefix][vector])))

    outfile.close()



    print timestamp(), "done"