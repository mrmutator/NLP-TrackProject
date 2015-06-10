__author__ = 'rwechsler'
import cPickle as pickle
import gensim
import itertools
import numpy as np

def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)



def evaluate_set(prefix, tails, word2vec_model):
    similarities = []
    for (fl1, tail1), (fl2, tail2) in itertools.permutations(tails, 2):
        try:
            diff = word2vec_model[prefix + fl2 + tail2] - word2vec_model[tail2]
            target = word2vec_model[prefix + fl1 + tail1]
            predicted = word2vec_model[tail1] + diff
            # cosine similarity
            similarities.append(np.dot(gensim.matutils.unitvec(target), gensim.matutils.unitvec(predicted)))

        except KeyError:
            pass

    return np.mean(similarities)


if __name__ == "__main__":

    combinations = load_candidate_dump("models/comb_model.p")
    word2vec_model = load_word2vecmodel("models/mono_200_de.bin")

    for k in combinations:
        print k
        print evaluate_set(k, combinations[k], word2vec_model)
        print "---"

