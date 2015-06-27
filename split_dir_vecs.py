__author__ = 'rwechsler'

import cPickle as pickle
import sys
import numpy as np
from collections import defaultdict

# argument1=dir_vec_pickle_file, argument2=number_of_splits

dir_vecs = pickle.load(open(sys.argv[1], "rb"))

length_tuples = sorted([(prefix, prototype, evidence_set) for prefix in dir_vecs for prototype, evidence_set in dir_vecs[prefix]], key=lambda t: len(t[2]), reverse=True)

print "Number of prototypes: ", len(length_tuples)

num_splits = int(sys.argv[2])

splits = [defaultdict(list) for i in range(num_splits)]
lengths = [0 for i in range(num_splits)]

# Simple heuristic

for prefix, prototype, evidence_set in length_tuples:
    target = np.argmin(lengths)
    splits[target][prefix].append((prototype, evidence_set))
    lengths[target] += len(evidence_set)

print "Length distribution:", lengths

for i, split in enumerate(splits):
    pickle.dump(split, open(sys.argv[1]+"."+str(i+1), "wb"))


