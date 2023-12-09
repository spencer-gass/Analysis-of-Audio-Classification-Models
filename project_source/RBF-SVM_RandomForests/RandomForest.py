import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
# from fileUtils import *
from ClassifierStateSaver import ClassifierStateSaver
from random import shuffle
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from operator import itemgetter


from os import listdir
from os.path import basename, join
import numpy as np
from time import time

from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def getFileDict(path):
    # get file names
    directory = listdir(path)
    files = [f for f in directory]

    # files can be divided into categories by the first two words of the file
    # base name
    # make a set of file name substrings
    fileBaseNames = [basename(f).partition('_')[0] for f in files]

    # make dictionary keys from set of fileBaseNames
    fileDict = {key: [] for key in set(fileBaseNames)}

    # fill in dict
    for index, file in enumerate(files):
        fileDict[fileBaseNames[index]].append(file)

    # return it as a list of lists
    return fileDict


"""grab only the first "elementsPerList" from each list in the
dict (each instrument)"""


def getSubsetOfFileDict(fileDict, elementsPerList):
    return {k: v[:elementsPerList] for (k, v) in fileDict.items()}


def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


def loadFiles(fileList, basePath):
    num_files = len(fileList)
    reshape_len = 256*63
    data = np.zeros(shape=(num_files, 256*63))
    labels = []
    index = 0
    for tup in fileList:
        data[index] = (np.load(file=join(basePath, tup[1])))[
            :256, :63].reshape(reshape_len)
        labels.append(tup[0])
        index += 1
    return labels, data


if __name__ == '__main__':
    path = './nsynth-train-tranformed1/audio'
    fileDict = getFileDict(path)

    classes = list(fileDict.keys())
    # generate classifier encoding
    le = preprocessing.LabelEncoder()
    le.fit(classes)

    fileTupleList = []
    totalSamples = recursive_len(list(fileDict.values()))
    #print(f"total Samples : {totalSamples}")
    totalClasses = len(list(fileDict.keys()))
    #print(f"total Classes : {totalClasses}")

    weightDict = {}
    for (k, v) in fileDict.items():
        weightDict[le.transform([k])[0]] = totalSamples / (len(v)*totalClasses)
        fileTupleList += [(k, file) for file in v]

    print(weightDict)
    # create classifier
    # using all defaults for now
    #feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
    #nystroem_approx_svm = pipeline.Pipeline(
        #[("feature_map", feature_map_nystroem), ('svm', svm.LinearSVC())])

    batchSize = 289205
    sample_sizes = [500]
    nystroem_times = []
    # nystroem_score = []

    # demo batching
    # n_iter = 5
    # for iteration in range(n_iter):
    count = 0
    index = 0
    shuffle(fileTupleList)
    # print(f"Starting iteration {iteration}")
    batchList = fileTupleList[index:index+batchSize]
    d = {}
    for x, y in batchList:
        d[x] = d.get(x, 0)+1
    print(d)
    print(len(d))
    # print(f"On Batch #{count})
    print("Loading Batch")
    start = time()
    batchLabels, batchData = loadFiles(batchList, path)
    print("took: " + str(time()-start) + " seconds")
    # X,Y inputs for sklearn
    print("Scaling Batch")
    start = time()
    X = batchData
    #X = preprocessing.scale(batchData)
    Y = le.transform(batchLabels)
    print("took: " + str(time()-start) + " seconds")

    for D in sample_sizes:
        clf = RandomForestClassifier(max_leaf_nodes = 20, max_depth = 5, min_weight_fraction_leaf = 0.1, n_estimators = 200, min_samples_split = 19, min_samples_leaf=9, class_weight = weightDict)
        print("Training on Batch")
        start = time()
        clf.fit(X, Y)
        # clf.partial_fit(X, Y, classes=le.transform(classes))
        #nystroem_times.append(time()-start)
        print("took: " + str(time()-start) + " seconds")
        index += batchSize
        count += 1
        clfSaver = ClassifierStateSaver(clf)

        fileName = "Random_Forest_Leaf_Correct_Way.bin"

        with open(fileName, 'wb') as fp:
            pickle.dump(clfSaver, fp)
