__author__ = 'rwechsler'


from corpus_reader import CorpusReader
import dawg
from collections import defaultdict


def build_vocabulary(corpus, min_length=0, fugenelemente=[]):

    prefix_vocab = set()
    suffix_vocab = set()
    for tokens in corpus:
        for tok in tokens:
            if len(tok) >= min_length:
                prefix_vocab.add(tok)
                suffix_vocab.add(tok[::-1])
                for fe in fugenelemente:
                    prefix_vocab.add(tok + fe)
                    suffix_vocab.add(tok + fe[::-1])


    return prefix_vocab, suffix_vocab

def add_prefix_combinations(combinations, prefix_vocab, dawg_model, fugenlaute=[""]):
    for word in prefix_vocab:
        for prefix in dawg_model.prefixes(word)[:-1]: # last word is the word itself
            rest = word[len(prefix):]
            # Consider fugenlaute
            for fl in fugenlaute:
                if rest.startswith(fl):
                    if rest.lstrip(fl).title() in dawg_model:
                        combinations[prefix].add((fl, rest.lstrip(fl).title()))
                    elif rest.lstrip(fl) in dawg_model:
                        combinations[prefix].add((fl, rest.lstrip(fl)))

    return

# def add_suffix_combinations(combinations, suffix_vocab, dawg_model, fugenlaute=[""]):
#     fugenlaute = [fl[::-1] for fl in fugenlaute]
#     for word in suffix_vocab:
#         for prefix in dawg_model.prefixes(word)[:-1]: # last word is the word itself
#             rest = word[len(prefix):]
#             # Consider fugenlaute
#             for fl in fugenlaute:
#                     if rest in dawg_model or rest.lstrip().title() in dawg_model:
#                         combinations[prefix].add((fl, rest.lstrip(fl)))
#
#     return

corpus = CorpusReader("data/news.2011.de.shuffled.gz", max_limit=1000000)

prefix_vocab, suffix_vocab = build_vocabulary(corpus, min_length=4)
dawg_model = dawg.DAWG(prefix_vocab)

print len(prefix_vocab)


fugenlaute = ["", "en", "s"] # make sure that empty string is always there

combinations = defaultdict(set)
add_prefix_combinations(combinations, prefix_vocab, dawg_model, fugenlaute=fugenlaute)


print len(combinations)

for k, v in combinations.items()[:40]:
    print k, v