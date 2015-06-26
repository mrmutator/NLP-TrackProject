__author__ = 'lqrz'

import cPickle as pickle
import gensim
import logging
import pdb
from nltk.corpus import PlaintextCorpusReader
import sys
import multiprocessing as mp
import codecs

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('')
hdlr = logging.FileHandler('decompound2.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def decompound((inputCompound, nAccuracy)):

    global model
    global vectors

    print id(model)

    if len(inputCompound) == 0:
        return []

    # get all matching prefixes
    logger.info('Getting all matching prefixes')
    prefixes = set()
    for prefix in vectors.keys():
        found = inputCompound.find(prefix)
        if found == 0 and len(vectors[prefix]) > 0 and len(inputCompound[len(prefix):])>0:
            prefixes.add(prefix)

    logger.debug('Possible prefixes')
    logger.debug(prefixes)

    # get all possible splits
    logger.info('Getting possible splits')

    splits = set()
    splitsWithNoRep = set()

    for prefix in prefixes:
        fugenlaute = ['', 'e', 'es']
        for fug in fugenlaute:
            if fug == '' or inputCompound[len(prefix):].find(fug) == 0:
                if not debug:
                    try:
                        # look for the uppercased rest representation
                        tail = inputCompound[len(prefix) + len(fug):].title()
                        tailRepresentationIndex = model.vocab[tail].index
                        splits.add((prefix, tail, tailRepresentationIndex))
                        msg = ' '.join(['Considering split', inputCompound, prefix, tail])
                        logger.debug(msg)
                    except KeyError:
                        # if i dont have a vector rep for the rest, i discard it
                        splitsWithNoRep.add((prefix, tail))
                        msg = ' '.join(['Discarding split', inputCompound, prefix, tail])
                        logger.debug(msg)
                    try:
                        # look for the lowercased rest representation
                        tail = inputCompound[len(prefix) + len(fug):]
                        tailRepresentationIndex = model.vocab[tail].index
                        splits.add((prefix, tail, tailRepresentationIndex))
                        msg = ' '.join(['Considering split', inputCompound, prefix, tail])
                        logger.debug(msg)
                    except KeyError:
                        # if i dont have a vector rep for the rest, i discard it
                        msg = ' '.join(['Discarding split', inputCompound, prefix, tail])
                        logger.debug(msg)
                        splitsWithNoRep.add((prefix, tail))
                        continue

    if len(splits) == 0:
        logger.error('Cannot decompound '+inputCompound)
        # exit()
        return [(inputCompound, '', '')]

    # apply direction vectors to splits
    logger.info('Applying direction vectors to possible splits')
    representations = set()
    bestSimilarity = 0.46 # so we do not split "Bahnhof" = ["Bahn", "Hof"]
    best = None
    maxEvidence = 0
    bestEvidence = None
    for prefix, tail, tailRepresentationIndex in splits:
        msg = ' '.join(['Applying', str(len(vectors[prefix])), 'direction vectors to split', prefix, tail])
        logger.debug(msg)
        for origin, evidence in vectors[prefix]:
            dirVectorCompoundRepresentation = model.syn0[origin[0]]
            dirVectorTailRepresentation = model.syn0[origin[1]]
            dirVectorDifference = dirVectorCompoundRepresentation - dirVectorTailRepresentation
            predictionRepresentation = model.syn0[tailRepresentationIndex] + dirVectorDifference

            # accuracy
            neighbours = sorted(model.most_similar(positive=[predictionRepresentation], negative=[], topn=nAccuracy), \
                                key=lambda x: x[1], reverse=True)
            try:
                metrics = [(i, t[1]) for i, t in enumerate(neighbours) if t[0] == inputCompound][0]
                res = (prefix, tail, origin[0], origin[1], metrics[0], metrics[1])
                representations.add(res)

                if metrics[1] > bestSimilarity: # compare cosine similarity
                    bestSimilarity = metrics[1]
                    best = res
            except IndexError:
                splitsWithNoRep.add((prefix, tail))
                res = (prefix, tail, origin[0], origin[1])
                if len(evidence) > maxEvidence:
                    maxEvidence = len(evidence)
                    bestEvidence = res

                continue

    chosenSplit = None

    if best:
        chosenSplit = best
        msg = ' '.join(['Splitting',inputCompound,'as', chosenSplit[0], chosenSplit[1], str(chosenSplit[2]), \
                        str(chosenSplit[3]), 'rank', str(chosenSplit[4]), 'similarity', str(chosenSplit[5])])
        logger.debug(msg)
        logger.debug('Decompounding '+chosenSplit[1])
    else:
        # nobody got the original representation within the KNN
        # chosenSplit = bestEvidence
        chosenSplit = (inputCompound, '') # not split at all
        logger.debug('Not splitting compound '+inputCompound)


    # logging.debug('Found prefix '+chosenSplit[0])
    # logging.debug('Decompounding '+chosenSplit[1])
    return [(inputCompound, chosenSplit[0], chosenSplit[1])] + decompound((chosenSplit[1], nAccuracy))


if __name__ == '__main__':
    resultsPath = 'results/dir_vecs_4_100.p'
    w2vPath = 'models/mono_500_de.bin'

    # inputCompounds = ['Hauptbahnhof', 'Frauenfilmfestival', 'Fussbodenschleifmaschinenverleih']

    corpusPath = './prueba.txt'

    outPath = 'splits.txt'

    multiprocessed = True
    nWorkers = 4

    if len(sys.argv) == 7:
        resultsPath = sys.argv[1]
        w2vPath = sys.argv[2]
        corpusPath = sys.argv[3]
        multiprocessed = sys.argv[4]
        nWorkers = sys.argv[5]
        outPath = sys.argv[6]

    elif len(sys.argv)>1:
        print 'Error in params'
        exit()

    idx = corpusPath.rfind('/') + 1
    folder = corpusPath[0:idx]
    filename = corpusPath[idx:]

    corpus = PlaintextCorpusReader(folder, filename, encoding='utf-8')
    inputCompounds = corpus.words()

    #TODO: define threshold
    nAccuracy= 30

    debug = False

    logger.info('Getting pickle results file')
    vectors = pickle.load(open(resultsPath, 'rb'))

    if not debug:
        logger.info('Getting word2vec model')
        model = gensim.models.Word2Vec.load_word2vec_format(w2vPath, binary=True)

    if multiprocessed:
        logger.info('Instantiating pool with '+str(nWorkers))
        pool = mp.Pool(processes=int(nWorkers))
        results = pool.map(decompound, zip(inputCompounds,[nAccuracy]*len(inputCompounds)))
    else:
        results = []
        for inputCompound in inputCompounds:
            if not debug:
                try:
                    compoundRepresentation = model[inputCompound]
                    results.append(decompound((inputCompound, nAccuracy)))
                except KeyError:
                    logger.error('No word2vec representation for input compound'+inputCompound)
                    # exit()
                    results.append(inputCompound)


    print results

    fout = codecs.open(outPath, 'w', encoding='utf-8')

    for i, split in enumerate(results):
        for comp, decomp1, decomp2 in split:
            fout.write(comp + '\t' + decomp1 + '\t' + decomp2 + '\n')

    fout.close()

    logger.info('End')