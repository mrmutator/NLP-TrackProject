__author__ = 'rwechsler'
import codecs
import sys
import numpy as np
from collections import defaultdict
infile = codecs.open(sys.argv[1], "r", "utf-8")

sums = np.array([0,0,0,0,0])

prefixes = defaultdict(list)

vectors = 0
for line in infile:
    vectors += 1
    els = line.strip().split("\t")
    result =  els[2:]
    prefixes[els[0]].append(result)
    sums += np.array(result)

infile.close()



results = sums / float(vectors)

print "Total results"

for prefix in prefixes:
    prefixes[prefix] = np.mean(prefixes[prefix], axis=0)


for prefix, r in sorted(prefix.items(), key=lambda t: t[1][0], reverse=True)[:20]:
    print prefix, r

    











