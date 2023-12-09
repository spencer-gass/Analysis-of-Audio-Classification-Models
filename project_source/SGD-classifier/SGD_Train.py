import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from fileUtils import *
from ClassifierStateSaver import ClassifierStateSaver
from random import shuffle
import time
import pickle


if __name__ == '__main__':
    path = './nsynth-transformed2/nsynth-transformed2/nsynth-train-tranformed2/audio'
    allFileDict = getFileDict(path)

    allClasses = list(allFileDict.keys())

    fileDict = {}
    classes = allClasses[:]
    for cl in classes:
        fileDict[cl] = allFileDict[cl]

    # generate classifier encoding
    le = preprocessing.LabelEncoder()
    le.fit(classes)

    fileTupleList = []
    totalSamples = recursive_len(list(fileDict.values()))
    print(f"total Samples : {totalSamples}")
    totalClasses = len(list(fileDict.keys()))
    print(f"total Classes : {totalClasses}")

    weightDict = {}
    for (k,v) in fileDict.items():
        weightDict[le.transform([k])[0]] = totalSamples/ (len(v)*totalClasses)
        fileTupleList+=[(k,file) for file in v]

    print(weightDict)
    #create classifier
    # using all defaults for now
    clf = SGDClassifier(class_weight= weightDict,tol = 1e-9)    

    batchSize = 80000

    # demo batching
    n_iter = 5
    for iteration in range(n_iter):
        count = 0
        index = 0
        shuffle(fileTupleList)
        print(f"Starting iteration {iteration}")
        lastBatch = False
        while not lastBatch:
            batchList = fileTupleList[index:index+batchSize]
            if len(batchList)<batchSize:
                lastBatch = True
                break
            print (f"On Batch #{count}")
            print("Loading Batch")
            batchLabels , batchData = loadFiles(batchList, path)
            # X,Y inputs for sklearn
            print("Scaling Batch")
            X = preprocessing.scale(batchData)
            Y = le.transform(batchLabels)
            print("Training on Batch")
            clf.partial_fit(X,Y, classes=le.transform(classes))
            index+=batchSize
            count+=1
    clfSaver = ClassifierStateSaver(clf)
    
    with open('SGD_Trained_Classifier.bin', 'wb') as fp:
        pickle.dump(clfSaver,fp)