__author__ = 'lqrz'
import gensim
import logging
from nltk.corpus import PlaintextCorpusReader
import pickle
import numpy as np
import sys


def decompound(inputCompound):
    global vectors
    global model
    global resultIndexes
    global resultVectors

    # get all matching prefixes
    logger.info('Getting all matching prefixes')
    prefixes = set()
    for prefix in vectors.keys():
        found = inputCompound.find(prefix)
        if found == 0 and len(vectors[prefix]) > 0 and len(inputCompound[len(prefix):]) > 0:
            prefixes.add(prefix)

    logger.debug('Possible prefixes')
    logger.debug(prefixes)

    # get all possible splits
    logger.info('Getting possible splits')

    splits = set()
    # splitsWithNoRep = set()

    fugenlaute = ['', 'e', 'es']

    for prefix in prefixes:
        for fug in fugenlaute:
            if fug == '' or inputCompound[len(prefix):].find(fug) == 0:
                try:
                    # look for the uppercased rest representation
                    tail = inputCompound[len(prefix) + len(fug):].title()
                    tailRepresentationIndex = model.vocab[tail].index
                    resultIndexes[tail] = tailRepresentationIndex
                    resultVectors[tailRepresentationIndex] = np.array(model[tail])
                    splits.add((prefix, tail, tailRepresentationIndex))
                    msg = ' '.join(['Considering split', inputCompound, prefix, tail])
                    logger.debug(msg)
                except KeyError:
                    # if i dont have a vector rep for the rest, i discard it
                    # splitsWithNoRep.add((prefix, tail))
                    msg = ' '.join(['Discarding split', inputCompound, prefix, tail])
                    logger.debug(msg)
                try:
                    # look for the lowercased rest representation
                    tail = inputCompound[len(prefix) + len(fug):]
                    tailRepresentationIndex = model.vocab[tail].index
                    resultIndexes[tail] = tailRepresentationIndex
                    resultVectors[tailRepresentationIndex] = np.array(model[tail])
                    splits.add((prefix, tail, tailRepresentationIndex))
                    msg = ' '.join(['Considering split', inputCompound, prefix, tail])
                    logger.debug(msg)
                except KeyError:
                    # if i dont have a vector rep for the rest, i discard it
                    msg = ' '.join(['Discarding split', inputCompound, prefix, tail])
                    logger.debug(msg)
                    # splitsWithNoRep.add((prefix, tail))
                    continue

    for prefix, tail, tailRepresentationIndex in splits:
        msg = ' '.join(['Applying', str(len(vectors[prefix])), 'direction vectors to split', prefix, tail])
        logger.debug(msg)
        for origin, evidence in vectors[prefix]:
            resultVectors[origin[0]] = np.array(model.syn0[origin[0]])
            resultVectors[origin[1]] = np.array(model.syn0[origin[1]])

    return True


if __name__ == '__main__':

    if len(sys.argv) == 5:
        resultsPath = sys.argv[1]
        corpusPath = sys.argv[2]
        w2vPath = sys.argv[3]
        resultsFolder = sys.argv[4]
    elif len(sys.argv) > 1:
        print 'Error in params'
        exit()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    # resultsPath = 'results/dir_vecs_4_100.p'
    # corpusPath = './prueba.txt'
    # outPath = 'splits.txt'
    # w2vPath = 'models/mono_500_de.bin'

    indexesPickleFilename = resultsFolder + 'decompoundIndexes.p'
    vectorsPickleFilename = resultsFolder + 'decompoundVectors.p'

    resultIndexes = dict()
    resultVectors = dict()

    logger.info('Getting pickle results file')
    vectors = pickle.load(open(resultsPath, 'rb'))

    model = gensim.models.Word2Vec.load_word2vec_format(w2vPath, binary=True)

    idx = corpusPath.rfind('/') + 1
    folder = corpusPath[0:idx]
    filename = corpusPath[idx:]

    corpus = PlaintextCorpusReader(folder, filename, encoding='utf-8')
    inputCompounds = corpus.words()

    # results = []
    for inputCompound in inputCompounds:
        try:
            compoundIndex = model.vocab[inputCompound].index
            # results.append(decompound(inputCompound))
            resultIndexes[inputCompound] = compoundIndex
            resultVectors[compoundIndex] = np.array(model[inputCompound])
            decompound(inputCompound)
        except KeyError:
            logger.error('No word2vec representation for input compound'+inputCompound)
            # exit()
            # results.append(inputCompound)
            continue

    logger.info('Pickling files')
    pickle.dump(resultIndexes, open(indexesPickleFilename,'wb'))
    pickle.dump(resultVectors, open(vectorsPickleFilename,'wb'))