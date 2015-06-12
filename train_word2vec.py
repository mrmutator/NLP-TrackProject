__author__ = 'rwechsler'

import gensim
import sys
import glob
from corpus_reader import CorpusReader



files = glob.glob(sys.argv[1])
outfile_name = sys.argv[2]

dataset = CorpusReader(files)

model = gensim.models.Word2Vec(dataset, size=500, window=5, min_count=3, negative=5, workers=2)


model.save(outfile_name)











