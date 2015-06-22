__author__ = 'lqrz'

import os
import sys
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pickle
import random
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
import gensim
from tsne import tsne
import numpy as Math
import pylab
import codecs

def plotGraph(samples, word, dimensions):
    if dimensions == '2D':
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        plt.plot(sklearn_transf[:,0],sklearn_transf[:,1],\
             'o', markersize=7, color='blue', alpha=0.5, label='')
        # plt.plot(sklearn_transf[1::2,0], sklearn_transf[1::2,1],\
        #      '^', markersize=7, color='red', alpha=0.5, label='Matrix')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    #     plt.xlim([-4,4])
        plt.ylim([-.8,.8])
        plt.legend()
        plt.title('Word embeddings PCA')

        print sklearn_transf

    elif dimensions == '3D':
        sklearn_pca = sklearnPCA(n_components=3)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        ax.plot(sklearn_transf[:,0], sklearn_transf[:,1],\
            sklearn_transf[:,2], 'o', markersize=8, color='blue', alpha=0.5, label='')
        # ax.plot(sklearn_transf[:,0], sklearn_transf[:,1],\
        #     sklearn_transf[:,2], '^', markersize=8, alpha=0.5, color='red', label='Matrix')

        plt.title('Word embeddings PCA')
        ax.legend(loc='upper right')

        print sklearn_transf

    plt.savefig("%s-%s.png" % (word, dimensions), bbox_inches='tight', dpi=200)
    plt.close()

    return True

def constructSamplesAndPlot(word1, tail1, word2, tail2, model):

    layerSize = model.layer1_size

    x = np.empty((0, layerSize))

    word1Vector = model[word1]
    tailVector1 = model[tail1]

    word2Vector = model[word2]
    tailVector2 = model[tail2]

    diferenceVector = word2Vector - tailVector2
    constructedVector = diferenceVector + tailVector1

    idx = word2.find(tail2.lower())
    if idx == -1:
        print 'Error.'
        exit()

    head2 = word2[:idx]
    head2Vector = model[head2]

    x = np.r_[x, word1Vector[np.newaxis,:]]
    x = np.r_[x, tailVector1[np.newaxis,:]]
    x = np.r_[x, word2Vector[np.newaxis,:]]
    x = np.r_[x, tailVector2[np.newaxis,:]]
    x = np.r_[x, constructedVector[np.newaxis,:]]
    x = np.r_[x, diferenceVector[np.newaxis,:]]
    x = np.r_[x, head2Vector[np.newaxis,:]]

    plotGraph(x, word1, dimensions='2D')
    plotGraph(x, word1, dimensions='3D')

    return True

def loadW2VModel(path):

    return gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

def plotPca():
    # n_components = 3
    modelPathDe = '../NLP2-Project2/models/mono_800_de.bin'
    # Load word2vec trained models
    print('Loading word2vec models...')
    model = loadW2VModel(modelPathDe)
    word1 = 'Hauptbahnhof'
    tail1 = 'Bahnhof'
    word2 = 'Hauptstadt'
    tail2 = 'Stadt'
    words = dict()
    words['word1'] = word1
    words['tail1'] = tail1
    words['word2'] = word2
    words['tail2'] = tail2
    words['model'] = model
    constructSamplesAndPlot(**words)


def plotTsne():
    w2vThreshold = 2
    filenames = ['Haupt.txt', 'Super.txt', 'Kinder.txt', 'Bundes.txt', 'Finanz.txt']
    # filenames = ['Haupt.txt', 'Bundes.txt']
    w2vPath = '../NLP2-Project2/models/mono_500_de.bin'
    # w2vPath = '../NLP2-Project2/models/mono_200_de.bin'
    dimensions = 500
    # dimensions = 200

    colours = ['#f02720', '#ff7f0f', '#32a251', '#1f77b4', '#ab6ad5']

    words = set()

    rawLabels = []

    for i, fname in enumerate(filenames):
        f = codecs.open(fname, 'rb', encoding='utf-8')
        for l in f:
            clean = l.strip().split(' ')
            if clean[0] > w2vThreshold:
                words.add(clean[1])
                rawLabels.append(colours[i])

    model = loadW2VModel(w2vPath)

    X = Math.empty((0, dimensions))
    # labels = Math.empty((1),dtype=float)

    labels = []

    for i,w in enumerate(words):
        try:
            rep = model[w]
            X = Math.r_[X, rep[Math.newaxis,:]]
            labels.append(rawLabels[i])
        except KeyError:
            continue

    # X = Math.loadtxt()
    # labels = Math.loadtxt()
    Y = tsne(X, 2, dimensions, 20.0, max_iter=1000)
    pylab.scatter(Y[:,0], Y[:,1], 18, marker='o', c=labels, edgecolor='None')
    pylab.savefig('scatter.png')
    # pylab.show()


def plotScatterPca():
    w2vThreshold = 2
    filenames = ['Arbeit.txt', 'Mann.txt', 'Ministerium.txt', 'Stadt.txt']
    # filenames = ['Haupt.txt', 'Bundes.txt']
    w2vPath = '../NLP2-Project2/models/mono_500_de.bin'
    # w2vPath = '../NLP2-Project2/models/mono_200_de.bin'
    dimensions = 500
    # dimensions = 200

    colours = ['#f02720', '#ff7f0f', '#32a251', '#1f77b4', '#ab6ad5']

    words = set()

    rawLabels = []

    for i, fname in enumerate(filenames):
        f = codecs.open(fname, 'rb', encoding='utf-8')
        for l in f:
            clean = l.strip().split(' ')
            if clean[0] > w2vThreshold:
                words.add(clean[1])
                rawLabels.append(colours[i])

    model = loadW2VModel(w2vPath)

    X = Math.empty((0, dimensions))
    # labels = Math.empty((1),dtype=float)

    labels = []

    for i,w in enumerate(words):
        try:
            rep = model[w]
            X = Math.r_[X, rep[Math.newaxis,:]]
            labels.append(rawLabels[i])
        except KeyError:
            continue

    # X = Math.loadtxt()
    # labels = Math.loadtxt()
    # Y = tsne(X, 2, dimensions, 20.0, max_iter=1000)

    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(X)

    # plt.plot(sklearn_transf[:,0],sklearn_transf[:,1],\
    #      'o', markersize=7, color='blue', alpha=0.5, label='')
    # plt.plot(sklearn_transf[1::2,0], sklearn_transf[1::2,1],\
    #      '^', markersize=7, color='red', alpha=0.5, label='Matrix')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
#     plt.xlim([-4,4])
    plt.ylim([-.8,.8])
    plt.legend()
    plt.title('Word embeddings PCA')

    pylab.scatter(sklearn_transf[:,0], sklearn_transf[:,1], 18, marker='o', c=labels, edgecolor='None')
    pylab.savefig('scatter.png')
    # pylab.show()

if __name__ == '__main__':
    # plotPca()

    # plotTsne()

    plotScatterPca()
