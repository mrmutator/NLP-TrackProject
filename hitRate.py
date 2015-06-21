__author__ = 'lqrz'

import pickle
import glob
import codecs

if __name__ == '__main__':
    fnames = glob.glob('/home/lquiroz/jobs/evaluate_candidates/output*/results.p')

    fout = codecs.open('hitRates.txt', 'w', 'utf-8')

    results = []
    for fPath in fnames:
        results.extend(pickle.load(open(fPath, 'r')))

    orderedResults = sorted([(k, r) for k,r in results], key=lambda t: t[1], reverse=True)
    
    for k,r in orderedResults:
        fout.write(k+'\t'+str(r)+'\n')

    fout.close()
