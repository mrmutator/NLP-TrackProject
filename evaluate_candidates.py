__author__ = 'rwechsler'
import cPickle as pickle
import gensim
import itertools
import numpy as np
from annoy import AnnoyIndex


def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

def build_annoy_tree(word2vec_model, output_file_name=None):
    tree = AnnoyIndex(word2vec_model.layer1_size)
    for i, word in enumerate(word2vec_model.index2word):
        tree.add_item(i, list(word2vec_model[word]))

    tree.build(10) #  how many?

    if output_file_name:
        tree.save(output_file_name)

    return tree

def load_annoy_tree(model_file_name, word2vec_model):
    tree = AnnoyIndex(word2vec_model.layer1_size)
    tree.load(model_file_name)
    return tree



def evaluate_set(prefix, tails, word2vec_model, annoy_tree, rank_threshold=100):
    # similarities = []
    positive_counts = 0
    negative_counts = 0
    for (fl1, tail1), (fl2, tail2) in itertools.permutations(tails, 2):
        try:
            diff = word2vec_model[prefix + fl2 + tail2] - word2vec_model[tail2]
            predicted = word2vec_model[tail1] + diff
            # cosine similarity
            # true_vector = word2vec_model[prefix + fl1 + tail1]
            # similarities.append(np.dot(gensim.matutils.unitvec(true_vector), gensim.matutils.unitvec(predicted)))

            true_index = word2vec_model.vocab[prefix + fl1 + tail1].index


            neighbours = annoy_tree.get_nns_by_vector(list(predicted), rank_threshold)

            if true_index in neighbours:
               positive_counts += 1
            else:
               negative_counts += 1

        except KeyError:
            pass

    return float(positive_counts) / (positive_counts + negative_counts) if positive_counts + negative_counts > 0 else 0


if __name__ == "__main__":

    print "candidates"
    candidates = load_candidate_dump("models/comb_model.p")
    print "loading word2vec model"
    word2vec_model = load_word2vecmodel("models/mono_500_de.model")
    print "building annoy tree"
    annoy_tree = build_annoy_tree(word2vec_model, output_file_name="tree.ann")
    # print "loading annoy tree"
    # annoy_tree = load_annoy_tree("test.ann", word2vec_model)

    results = dict()
    for k in candidates:
        # print k
        # print evaluate_set(k, combinations[k], word2vec_model, annoy_tree)
        # print "---"

        print k.encode("utf-8")
        result = evaluate_set(k, candidates[k], word2vec_model, annoy_tree, rank_threshold=100)
        if result > 0:
            results[k] = result

    print "pickling"
    pickle.dump(open("results.p", "wb"))

