__author__ = 'lqrz'
import codecs
import logging
import glob
import sys

logger = logging.getLogger('')
hdlr = logging.FileHandler('decompoundingEvaluation.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    # goldFile = 'decompounding/lqrz_compounds.test' # has to start with 2 lines of header
    # # resultsFiles = glob.glob('/home/lquiroz/jobs/decompound/100_046/output*/results.txt')
    # resultsFiles = glob.glob('output*/results.txt')
    # resultsFolder = 'decompounding' # without last /

    if len(sys.argv) == 3:
        goldFile = sys.argv[1]
        resultsFolder = sys.argv[2]
    elif len(sys.argv) > 1:
        logger.error('Bad params.')
        exit()

    resultsFiles = glob.glob(resultsFolder+'/output*/results.txt')

    logger.debug('Nr of result files: '+str(len(resultsFiles)))

    goldCompounds = dict()
    resultsCompounds = dict()

    accuracy = 0

    coverage = 0

    noSplitsAtAll = 0
    discardedSplits = 0

    fgold = codecs.open(goldFile, 'r', encoding='utf-8')

    totalResults = 0
    for resultsFile in resultsFiles:
        fresults = codecs.open(resultsFile, 'r', encoding='utf-8')

        for l in fresults:
            cleanLine = l.strip('\n').split('\t')
            resultSplit = ' '.join([cleanLine[1], cleanLine[2]]).strip()
            resultsCompounds[cleanLine[0]] = resultSplit
            if cleanLine[1] == '' and cleanLine[2] == '':
                # We didnt find proper prefixes (either no prefix matched or the tails were not in the w2v model)
                noSplitsAtAll += 1
                continue
            elif cleanLine[0] == cleanLine[1] and cleanLine[2] == '':
                # We found possible splits, but they didnt pass rank and similarity thresholds
                discardedSplits += 1
                continue

            coverage += 1
            totalResults += 1

    totalGold = 0
    lineNr = 0
    for l in fgold:
        lineNr += 1
        # if lineNr < 3:
        if lineNr < 40:
            continue
        goldSplit = set()
        idx = l.strip('\n').find('|')
        if idx <>-1:
            # multiple possibilities
            compound = l.strip('\n').split('\t')[0]
            modifier1 = l.strip('\n').split('\t')[1].split('|')[0]
            modifier2 = l.strip('\n').split('\t')[1].split('|')[1].title() # the 2nd modifier appears lowercased.\
                                                                            #  Should be upppercase
            head = l.strip('\n').split('\t')[2]
            goldSplit.add(' '.join([modifier1, head]))
            goldSplit.add(' '.join([modifier2, head]))
        else:
            compound = l.strip('\n').split('\t')[0]
            goldSplit.add(' '.join([l.strip('\n').split('\t')[1],l.strip('\n').split('\t')[2]]))


        if resultsCompounds[compound] in goldSplit:
            accuracy += 1

        totalGold += 1

    assert totalResults == totalGold, 'Total nr of lines in gold file does not match total nr lines in results file'

    # Stats
    logger.info('Total number of examples: '+str(totalGold))
    logger.info('Coverage: '+str(coverage)+' '+str(coverage/float(totalGold)))
    logger.info('Examples for which no splits were found: '+str(noSplitsAtAll)+' '+\
                str(noSplitsAtAll/float(totalGold)))
    logger.info('Examples for which weak prefixes were found: '+str(discardedSplits)+' '+\
                str(discardedSplits/float((totalGold))))
    logger.info('Accuracy: '+str(accuracy/float(totalGold)))

    logger.info('End')