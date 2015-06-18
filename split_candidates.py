import cPickle as pickle
import sys
import numpy as np

# argument1=candidate_index_file, argument2=number_of_splits

candidates = pickle.load(open(sys.argv[1], "rb"))

length_tuples = sorted([(c, len(candidates[c])) for c in candidates if len(candidates[c] > 1)], key=lambda t: t[1], reverse=True)

print len(length_tuples)

num_splits = int(sys.argv[2])

splits = [{} for i in range(num_splits)]
lengths = [0 for i in range(num_splits)]

# Simple heuristic

for prefix, length in length_tuples:
    target = np.argmin(lengths)
    splits[target][prefix] = candidates[prefix]
    lengths[target] += length

print lengths

for i, split in enumerate(splits):
    pickle.dump(split, open(sys.argv[1]+"."+str(i+1)), "wb")
