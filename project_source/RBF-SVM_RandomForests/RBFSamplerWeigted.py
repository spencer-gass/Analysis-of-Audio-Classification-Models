import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
# from fileUtils import *
from ClassifierStateSaver import ClassifierStateSaver
from random import shuffle
import time
import pickle

from os import listdir
from os.path import basename, join
import numpy as np
from time import time

from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


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
    reshape_len = 20*63
    data = np.zeros(shape=(num_files, 20*63))
    labels = []
    index = 0
    for tup in fileList:
        data[index] = (np.load(file=join(basePath, tup[1])))[
            :20, :63].reshape(reshape_len)
        labels.append(tup[0])
        index += 1
    return labels, data


if __name__ == '__main__':
    path = './nsynth-train-tranformed2/audio'
    fileDict = getFileDict(path)

    classes = list(fileDict.keys())
    # generate classifier encoding
    le = preprocessing.LabelEncoder()
    le.fit(classes)

    fileTupleList = []
    totalSamples = recursive_len(list(fileDict.values()))
    print("total Samples : " +str (totalSamples)) 
    totalClasses = len(list(fileDict.keys()))
    #print(f"total Classes : {totalClasses}")

    weightDict = {}
    for (k, v) in fileDict.items():
        weightDict[le.transform([k])[0]] = totalSamples / (len(v)*totalClasses)
        fileTupleList += [(k, file) for file in v]

    print(weightDict)
    # create classifier
    # using all defaults for now
    feature_map_fourier = RBFSampler(gamma=.000139, random_state=1)
    fourier_approx_svm = pipeline.Pipeline(
        [("feature_map", feature_map_fourier), ('svm', svm.LinearSVC(class_weight=weightDict, C=26.82))])

    batchSize = 289205
    sample_sizes = [1000]
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
    # print(f"On Batch #{count})
    print("Loading Batch")
    start = time()
    batchLabels, batchData = loadFiles(batchList, path)
    print("took: " + str(time()-start) + " seconds")
    # X,Y inputs for sklearn
    print("Scaling Batch")
    start = time()
    X = preprocessing.scale(batchData)
    Y = le.transform(batchLabels)
    print("took: " + str(time()-start) + " seconds")

    for D in sample_sizes:
        fourier_approx_svm.set_params(feature_map__n_components=D)
        print("Training on Batch")
        start = time()
        fourier_approx_svm.fit(X, Y)
        # clf.partial_fit(X, Y, classes=le.transform(classes))
        #fourier_times.append(time()-start)
        print("took: " + str(time()-start) + " seconds")
        index += batchSize
        count += 1
        clfSaver = ClassifierStateSaver(fourier_approx_svm)

        fileName = "RBFSampler_Weighted_" + str(D) + ".bin"

        with open(fileName, 'wb') as fp:
            pickle.dump(clfSaver, fp)

