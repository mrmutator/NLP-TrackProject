__author__ = 'rwechsler'
import codecs
import gensim
import re
import numpy as np
from scipy.stats import spearmanr

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
        try:
            return word2vec_model[obj], 0
        except KeyError:
            return None, 0
    elif isinstance(obj, tuple):
        try:
            prefix = obj[0]
            tail = obj[1]
            protoype = prototypes[prefix]
            pro_compound = prefix + protoype[0] + protoype[1].lower()
            diff = word2vec_model[pro_compound] - word2vec_model[protoype[1]]
            compound_vec = word2vec_model[tail] + diff
            return compound_vec, 1
        except KeyError:
            return None, 0

def read_gold_sim(file_name):
    values = []
    infile = codecs.open(file_name, "r", "utf-8")
    for line in infile:
        els = line.strip().split(";")
        values.append(els[2])

    return values


if __name__ == "__main__":

    baseline = False

    word2vec_model = load_word2vecmodel("mono_500_de.bin")
    prototypes = load_prototypes("prototypes_6_100.txt")

    infile = codecs.open("splitting_results.txt","r", "utf-8")
    outfile = codecs.open("similarity_output.txt", "w", "utf-8")

    elements = []
    c = 0


    if baseline:
        for line in infile:
            els = line.strip().split("\t")
            elements.append(els[0],)
            c += 1
    else:
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

    annotated = read_gold_sim("gold.txt")

    counts = []
    sims = []
    gold = []
    c = 0
    for pair in pairs:
        w1 = pair[0]
        w2 = pair[1]
        vec1, i = get_vector(w1, word2vec_model, prototypes)
        vec2, j = get_vector(w2, word2vec_model, prototypes)

        if isinstance(w1, unicode):
            w1 = (w1,)
        if isinstance(w2, unicode):
            w2 = (w2,)

        if vec1 is not None and vec2 is not None:
            sim = np.dot(gensim.matutils.unitvec(vec1), gensim.matutils.unitvec(vec2))
            counts.append(i + j)
            outfile.write(";".join(["|".join(w1), "|".join(w2), str(sim)]) + "\n")
            sims.append(sim)
            gold.append(annotated[c])
        else:
            outfile.write(";".join(["|".join(w1), "|".join(w2), "NA"]) + "\n")

        c += 1


    outfile.write("------------\n")
    outfile.write("Total: " + str(sum(counts)) + "\n")
    outfile.write("Both: " + str(counts.count(2)) + "\n")
    outfile.write("Single: " + str(counts.count(1)) + "\n")
    outfile.write("Pairs used: " + str(len(sims)))

    s, p = spearmanr(sims, gold)

    outfile.write("Spearman R: " + str(s) + "\n")
    outfile.write("P: " + str(p) + "\n")
    outfile.close()