import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from fileUtils import *
from ClassifierStateSaver import ClassifierStateSaver 
import time
import pickle


if __name__ == '__main__':
    path = './nsynth-train-tranformed1/audio'
    fileDict = getFileDict(path)
    classes = list(fileDict.keys())

    # generate classifier encoding
    le = preprocessing.LabelEncoder()
    le.fit(classes)

    #create classifier
    with open('SGD_Trained_Classifier.bin', 'rb') as fp:
        clf = pickle.load(fp).clf    

    fileTupleList = []
    for (k,v) in fileDict.items():
        fileTupleList+=[(k,file) for file in v[:5]]
    print("Loading Batch")
    batchLabels , batchData = loadFiles(fileTupleList, path)
    # X,Y inputs for sklearn
    print("Scaling Batch")
    X = preprocessing.scale(batchData)
    Y = le.transform(batchLabels)
    prediction = clf.predict(X)
    print(list(le.inverse_transform(Y)))
    print(list(le.inverse_transform(prediction)))

