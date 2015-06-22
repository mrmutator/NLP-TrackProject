__author__ = 'rwechsler'
import gensim
import cPickle as pickle
import argparse
import sys
import codecs


def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)


def load_prototype_dump(file_name):
    return pickle.load(open(file_name, "rb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate candidates')
    parser.add_argument('-w', action='store', dest="word2vec_file", required=True)
    parser.add_argument('-p', action='store', dest='prototype_file', required=True)
    parser.add_argument('-o', action="store", dest='output_file', required=True)

    arguments = parser.parse_args(sys.argv[1:])

    word2vec_model = load_word2vecmodel(arguments.word2vec_file)
    prototype_set = load_prototype_dump(arguments.prototype_file)

    outfile = codecs.open(arguments.output_file, "w", "utf-8")

    for candidate in prototype_set:
        print candidate
        for prototype, evidence_set in candidate[1]:
            outfile.write(candidate[0] + "\t" + word2vec_model.index2word[prototype] + "\t" + " ".join([word2vec_model.index2word[i] for i in evidence_set]) + "\n")

    outfile.close()

