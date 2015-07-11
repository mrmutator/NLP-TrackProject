__author__ = 'rwechsler'
import codecs
from collections import defaultdict
import gensim
import re

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

def load_prototypes(file_name):
    prototypes = dict()
    infile = codecs.open(file_name, "r", "utf-8")
    for line in infile:
        els = line.split("\t")
        try:
            fl = re.search("\[(.*?)\]", els[1]).group(1)
        except AttributeError:
            fl = ""
        prototypes[els[0]] = (fl, re.sub("\[.*?\]", "", els[1]))


    infile.close()
    return prototypes


def get_vector(obj, word2vec_model, prototypes):
    if isinstance(obj, unicode):
        return word2vec_model[obj], 0
    elif isinstance(obj, tuple)
        prefix = obj[0]
        tail = obj[1]
        protoype = prototypes[prefix]
        pro_compound = prefix + protoype[0] + protoype[1].lower()
        diff = word2vec_model[pro_compound] - word2vec_model[protoype[1]]
        compound_vec = word2vec_model[tail] + diff
        return compound_vec, 1


if __name__ == "__main__":

    word2vec_model = load_word2vecmodel("test/mini_500_de.bin")
    prototypes = load_prototypes("data/prototypes_6_100.txt")

    infile = codecs.open("data/splitting/100_46_results.txt","r", "utf-8")
    outfile = codecs.open("data/splitting/output.txt", "w", "utf-8")

    elements = []
    c = 0
    for line in infile:
        els = line.strip().split("\t")
        if len(els) > 2:
            elements.append(tuple(els[1:]))
        else:
            elements.append(els[0],)
        c += 1


    infile.close()

    length = len(elements)

    pairs = zip(elements[:length/2], elements[length/2:])



    for pair in pairs:
        vec1 = get_vector(pair[0], word2vec_model, prototypes)
        vec2 = get_vector(pair[1], word2vec_model, prototypes)
        sim = 

