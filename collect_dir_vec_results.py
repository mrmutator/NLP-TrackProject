__author__ = 'lqrz'

import pickle
import glob
import codecs

if __name__ == '__main__':
    fnames = glob.glob('/home/lquiroz/jobs/find_direction_vectors/output*/results_dirvec.p')

    fout = codecs.open('dir_vecs.txt', 'w', 'utf-8')

    results = dict()
    for fPath in fnames:
        temp_result = pickle.load(open(fPath, 'r'))
        for entry in temp_result:
            results[entry[0]] = entry[1]



    pickle.dump(results, open("dir_vecs.p", "wb"))

    orderedResults = sorted([(k, len(results[k])) for k in results], key=lambda t: t[1], reverse=True)
    
    for k,r in orderedResults:
        fout.write(k+'\t'+str(r)+'\n')

    fout.close()
