__author__ = 'lqrz'

from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import codecs
import sys

if __name__ == '__main__':
    # corpus_path_train = '../NLP2-Project2/corpus/truecase/5k/'
    # corpus_path_test = '../NLP2-Project2/corpus/testing/'

    if len(sys.argv) == 3:
        corpus_path_train = sys.argv[1]
        corpus_path_test = sys.argv[2]
    elif len(sys.argv) > 1:
        print 'Error in params'
        exit()

    idx = corpus_path_train.rfind('/') + 1
    folder_train = corpus_path_train[0:idx]
    filename_train = corpus_path_train[idx:]

    idx = corpus_path_test.rfind('/') + 1
    folder_test = corpus_path_test[0:idx]
    filename_test = corpus_path_test[idx:]


    'Getting corpora'
    corpus_train = PlaintextCorpusReader(folder_train, filename_train, encoding='utf-8')
    corpus_test = PlaintextCorpusReader(folder_test, filename_test, encoding='utf-8')

    print 'Getting frequency distributions'
    fd_train = FreqDist([w for w in corpus_train.words() if w.isalpha()])
    fd_test = FreqDist([w for w in corpus_test.words() if w.isalpha()])

    # cross = sorted([(w,fd_train[w]) for w in fd_test.keys()], key=lambda x:x[1], reverse=True)

    thresholds = [0, 20, 50, 100, 500]

    for th in thresholds:

        print 'Processing th: ' + str(th)

        fout = codecs.open('MT/words_threshold_'+str(th), 'w', encoding='utf-8')

        for word in fd_test.keys():
            if fd_train[word] <= th:
                fout.write(word + '\t' + str(fd_train[word]) + '\n')

        fout.close()

    print 'End'