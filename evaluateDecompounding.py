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
    # # # resultsFiles = glob.glob('/home/lquiroz/jobs/decompound/100_046/output*/results.txt')
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

    noInputRepresentation = 0
    noTailRepresentation = 0
    noSplitsAtAll = 0
    discardedSplits = 0

    fgold = codecs.open(goldFile, 'r', encoding='utf-8')

    totalResults = 0
    for resultsFile in resultsFiles:
        fresults = codecs.open(resultsFile, 'r', encoding='utf-8')

        for l in fresults:
            totalResults += 1
            cleanLine = l.strip('\n').split('\t')

            if cleanLine[1] == 'Noinputrep':
                # We didnt find a representation for the compound
                resultsCompounds[cleanLine[0]] = ''
                noInputRepresentation += 1
                noSplitsAtAll += 1
                continue
            elif cleanLine[1] == 'Notailrep':
                # We didnt find a representation for the substring
                resultsCompounds[cleanLine[0]] = ''
                noTailRepresentation += 1
                noSplitsAtAll +=1
                continue
            elif cleanLine[0] == cleanLine[1] and cleanLine[2] == '':
                # We found possible splits, but they didnt pass rank and similarity thresholds
                resultsCompounds[cleanLine[0]] = ''
                discardedSplits += 1
                continue

            # We were able to split
            resultSplit = ' '.join([cleanLine[1], cleanLine[2]]).strip()
            resultsCompounds[cleanLine[0]] = resultSplit
            coverage += 1

    lineNr = 0
    for l in fgold:
        lineNr += 1
        # Gold file must have 2 header lines
        if lineNr < 3:
            continue
        goldSplit = set()
        idx = l.strip('\n').find('|')
        if idx != -1:
            # multiple possibilities
            compound = l.strip('\n').split('\t')[0].strip()
            modifier1 = l.strip('\n').split('\t')[1].split('|')[0].strip()
            modifier2 = l.strip('\n').split('\t')[1].split('|')[1].strip().title() # the 2nd modifier appears lowercased.\
                                                                            #  Should be upppercase
            head = l.strip('\n').split('\t')[2].strip()
            goldSplit.add(' '.join([modifier1, head]))
            goldSplit.add(' '.join([modifier2, head]))
        else:
            compound = l.strip('\n').split('\t')[0].strip()
            goldSplit.add(' '.join([l.strip('\n').split('\t')[1].strip(),l.strip('\n').split('\t')[2].strip()]))


        if resultsCompounds[compound] in goldSplit:
            accuracy += 1


    assert lineNr-2 == totalResults, 'Total nr of lines in gold file does not match total nr lines in results file '+\
        str(lineNr-2)+' '+str(totalResults)

    assert noSplitsAtAll == (noTailRepresentation+noInputRepresentation), 'Error in nr of no splits.'

    # Stats
    logger.info('Total number of examples: '+str(lineNr-2))
    logger.info('Coverage: '+str(coverage)+' '+str(coverage/float(lineNr-2))) # Measured against all word in gold file.

    logger.info('Examples for which no splits were found: '+str(noSplitsAtAll)+' '+\
                str(noSplitsAtAll/float(lineNr-2)))
    logger.info('No input representation found: '+str(noInputRepresentation)+' '+\
                str(noSplitsAtAll/float(lineNr-2)))
    logger.info('No tail representation found: '+str(noTailRepresentation)+' '+\
                str(noSplitsAtAll/float(lineNr-2)))

    logger.info('Examples for which weak prefixes were found: '+str(discardedSplits)+' '+\
                str(discardedSplits/float((coverage)))) # Measured against split compounds.
    logger.info('Accuracy: '+str(accuracy/float(coverage))) # Measured against split compounds.

    logger.info('End')