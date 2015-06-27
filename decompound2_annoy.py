__author__ = 'lqrz'

import cPickle as pickle
import gensim
import logging
import pdb
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import WhitespaceTokenizer
import sys
import multiprocessing as mp
import codecs
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('')
hdlr = logging.FileHandler('decompound2.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def decompound((inputCompound, nAccuracy)):

    global annoy_tree
    global vectors
    global pickledIndexes
    global pickledVectors


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
                        logger.debug('Tail: '+tail)
                        tailRepresentationIndex = pickledIndexes[tail]
                        logger.debug('Tail index: '+str(tailRepresentationIndex))
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
                        logger.debug('Tail: '+tail)
                        tailRepresentationIndex = pickledIndexes[tail]
                        logger.debug('Tail index: '+str(tailRepresentationIndex))
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
            logger.debug('Prefix '+prefix+' by indexes '+str(origin[0])+' and '+str(origin[1]))
            try:
                dirVectorCompoundRepresentation = pickledVectors[origin[0]]
                logger.debug('Found key in vector dict for index '+str(origin[0]))
            except KeyError:
                logger.debug('ERROR COULDNT FIND KEY '+str(origin[0])+' IN VECTOR DICT')
                continue

            try:
                dirVectorTailRepresentation = pickledVectors[origin[1]]
                logger.debug('Found key in vector dict for index '+str(origin[1]))
            except KeyError:
                logger.debug('ERROR COULDNT FIND KEY '+str(origin[1])+' IN VECTOR DICT')
                continue

            dirVectorDifference = dirVectorCompoundRepresentation - dirVectorTailRepresentation

            try:
                logger.debug('Looking up tail index '+str(tailRepresentationIndex))
                predictionRepresentation = pickledVectors[tailRepresentationIndex] + dirVectorDifference
                logger.debug('Found key in vector dict for index '+str(tailRepresentationIndex))
            except KeyError:
                logger.debug('ERROR COULDNT FIND KEY '+str(tailRepresentationIndex)+' IN VECTOR DICT')
                continue

            # accuracy
            # neighbours = sorted(model.most_similar(positive=[predictionRepresentation], negative=[], topn=nAccuracy), \
            #                     key=lambda x: x[1], reverse=True)

            logger.debug('Getting Annoy KNN')
            try:
                neighbours = annoy_tree.get_nns_by_vector(list(predictionRepresentation), nAccuracy)
                logger.debug(neighbours)
            except:
                logger.error('Problem found when retrieving KNN for prediction representation')
                logger.error(list(predictionRepresentation))
                exit()

            try:
                logger.debug('Looking up word '+inputCompound+' in index dict')
                inputCompoundIndex = pickledIndexes[inputCompound]
                logger.debug('Found key in index dict for word '+inputCompound)
            except KeyError:
                logger.debug('ERROR COULDNT FIND KEY '+inputCompound+' IN INDEX VECTOR')
                continue

            try:
                logger.debug('Looking up index '+str(inputCompoundIndex))
                inputCompoundRep = pickledVectors[inputCompoundIndex]
                logger.debug('Found key in vector dict for index '+str(inputCompoundIndex))
            except KeyError:
                logger.debug('ERROR COULDNT FIND KEY '+str(inputCompoundIndex)+' IN VECTOR DICT')
                continue

            try:
                rank = [i for i, nei in enumerate(neighbours) if nei == inputCompoundIndex][0]
                logger.debug(str(inputCompoundIndex)+' found in neighbours. Rank: '+str(rank))
                similarity = cosine_similarity(predictionRepresentation, inputCompoundRep)[0][0]
                logger.debug('Computed cosine similarity: '+str(similarity))
                res = (prefix, tail, origin[0], origin[1], rank, similarity)
                representations.add(res)
                if similarity > bestSimilarity: # compare cosine similarity
                    logger.debug('Found new best similarity score. Old: '+str(bestSimilarity)+' New: '+str(similarity))
                    bestSimilarity = similarity
                    best = res
            except IndexError:
                logger.debug(str(inputCompoundIndex)+' not found in neighbours. NO RANK. WONT SPLIT')
                continue
            # except IndexError:
            #     splitsWithNoRep.add((prefix, tail))
            #     res = (prefix, tail, origin[0], origin[1])
            #     if len(evidence) > maxEvidence:
            #         maxEvidence = len(evidence)
            #         bestEvidence = res
            #
            #     continue

    logger.debug('Choosing best direction vector')
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

    # return [(inputCompound, chosenSplit[0], chosenSplit[1])] + decompound((chosenSplit[1], nAccuracy))
    return [(inputCompound, chosenSplit[0], chosenSplit[1])] # do not apply recursion


if __name__ == '__main__':
    # resultsPath = 'results/dir_vecs_4_100.p'
    # annoyTreeFile = 'tree.ann'
    # pickledIndexes = pickle.load(open('decompoundIndexes.p','rb'))
    # pickledVectors = pickle.load(open('decompoundVectors.p','rb'))
    # corpusPath = './prueba.txt'
    # outPath = 'splits.txt'

    # multiprocessed = True
    # nWorkers = 4

    if len(sys.argv) == 9:
        resultsPath = sys.argv[1]
        # w2vPath = sys.argv[2]
        annoyTreeFile = sys.argv[2]
        corpusPath = sys.argv[3]
        pickledIndexesPath = sys.argv[4]
        pickledVectorsPath = sys.argv[5]
        multiprocessed = sys.argv[6]
        nWorkers = sys.argv[7]
        outPath = sys.argv[8]

    elif len(sys.argv)>1:
        print 'Error in params'
        exit()

    idx = corpusPath.rfind('/') + 1
    folder = corpusPath[0:idx]
    filename = corpusPath[idx:]

    logger.debug('Corpus folder: '+folder)
    logger.debug('Corpus filename: '+filename)

    corpus = PlaintextCorpusReader(folder, filename, word_tokenizer=WhitespaceTokenizer(), encoding='utf-8')
    inputCompounds = corpus.words()

    logger.debug('Words in corpus')
    logger.debug(inputCompounds)

    #TODO: define threshold
    nAccuracy= 100

    debug = False

    logger.info('Getting pickled direction vectors file')
    vectors = pickle.load(open(resultsPath, 'rb'))

    logger.info('Getting pickled indexes')
    pickledIndexes = pickle.load(open(pickledIndexesPath,'rb'))
    pickledVectors = pickle.load(open(pickledVectorsPath,'rb'))

    logger.info('Getting annoy tree')
    # model = gensim.models.Word2Vec.load_word2vec_format(w2vPath, binary=True)
    annoy_tree = AnnoyIndex(500)
    annoy_tree.load(annoyTreeFile)

    if multiprocessed:
        logger.info('Instantiating pool with '+str(nWorkers))
        pool = mp.Pool(processes=int(nWorkers))
        results = pool.map(decompound, zip(inputCompounds, [nAccuracy]*len(inputCompounds)))
    else:
        results = []
        for inputCompound in inputCompounds:
            try:
                inputCompoundIndex = pickledIndexes[inputCompound]
                compoundRepresentation = pickledVectors[inputCompoundIndex]
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