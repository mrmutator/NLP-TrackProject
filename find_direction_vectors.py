__author__ = 'rwechsler'
import datetime
import cPickle as pickle
from annoy import AnnoyIndex
import gensim
import time
import argparse
import sys
import numpy as np
import random
import multiprocessing as mp


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_annoy_tree(model_file_name, vector_dims):
    tree = AnnoyIndex(vector_dims)
    tree.load(model_file_name)
    return tree

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

def annoy_knn(annoy_tree, vector, true_index, k=100):
    neighbours = annoy_tree.get_nns_by_vector(list(vector), k)
    if true_index in neighbours:
        return True
    else:
        return False


def candidate_generator(candidates, rank_threshold, evidence_threshold):
    for prefix in candidates:
        yield (prefix, candidates[prefix], rank_threshold, evidence_threshold)

if __name__ == "__main__":


    #### Default Parameters-------------------------------------------####
    rank_threshold = 30
    evidence_threshold = 10
    sample_set_size = 500
    n_processes = 2
    ####End-Parametes-------------------------------------------------####


    parser = argparse.ArgumentParser(description='Evaluate candidates')
    parser.add_argument('-w', action='store', dest="word2vec_file", required=True)
    parser.add_argument('-d', action="store", dest="vector_dims", type=int, required=True)
    parser.add_argument('-t', action="store", dest="annoy_tree_file", required=True)
    parser.add_argument('-c', action="store", dest="candidates_index_file", required=True)
    parser.add_argument('-o', action="store", dest="result_output_file", required=True)
    parser.add_argument('-p', action="store", dest="n_processes", type=int, default=n_processes)
    parser.add_argument('-s', action="store", dest="sample_set_size", type=int, default=sample_set_size)
    parser.add_argument('-r', action="store", dest="rank_threshold", type=int, default=rank_threshold)
    parser.add_argument("-e", action="store", dest="evidence_threshold", type=int, default=evidence_threshold)

    arguments = parser.parse_args(sys.argv[1:])


    print timestamp(), "loading candidates"
    candidates = load_candidate_dump(arguments.candidates_index_file)

    print timestamp(), "loading word2vec model"
    word2vec_model = load_word2vecmodel(arguments.word2vec_file)

    print timestamp(), "preprocess candidates"
    # only store vectors that we need. And sample already.
    word2vec_vectors = dict()
    for cand in candidates:
        if len(candidates[cand]) > arguments.sample_set_size:
            candidates[cand] = set(random.sample(candidates[cand], arguments.sample_set_size))
        for (i,j) in candidates[cand]:
            word2vec_vectors[i] = np.array(word2vec_model.syn0[i])
            word2vec_vectors[j] = np.array(word2vec_model.syn0[j])

    del word2vec_model

    print timestamp(), "number of vectors: ", len(word2vec_vectors)

    print timestamp(), "load annoy tree"
    # global annoy_tree
    annoy_tree = load_annoy_tree(arguments.annoy_tree_file, arguments.vector_dims)

    def find_direction_vectors(prefix, tails, rank_threshold=100, evidence_threshold=10):
        global annoy_tree
        global word2vec_vectors

        evidence = dict()
        for (comp1, tail1) in tails:
            evidence[(comp1, tail1)] = set()
            diff = word2vec_vectors[comp1]- word2vec_vectors[tail1]
            for (comp2, tail2) in tails:
                predicted = word2vec_vectors[tail2] + diff
                result = annoy_knn(annoy_tree, predicted, comp2, rank_threshold)
                if result:
                    evidence[(comp1, tail1)].add((comp2, tail2))

        direction_vectors = []
        bool_continue = True
        while bool_continue:
            bool_continue = False
            # find best vector
            best_comp_pair = max(evidence, key=lambda k: len(evidence[k]))
            if len(evidence[best_comp_pair]) >= evidence_threshold:
                direction_vectors.append((best_comp_pair, set(evidence[best_comp_pair])))
                bool_continue = True

                # remove evidence
                best_evidence = set(evidence[best_comp_pair])
                for comp in evidence:
                    evidence[comp] = evidence[comp] - best_evidence
                del evidence[best_comp_pair]

        return (prefix, direction_vectors)

    def mp_wrapper_evaluate_set(argument):
        return find_direction_vectors(*argument)

    print timestamp(), "evaluating direction vectors"
    pool = mp.Pool(processes=arguments.n_processes)
    params = candidate_generator(candidates, arguments.rank_threshold, arguments.evidence_threshold)
    results = pool.map(mp_wrapper_evaluate_set, params)

    print timestamp(), "pickling"
    pickle.dump(results, open(arguments.result_output_file, "wb"))

    print timestamp(), "done"
