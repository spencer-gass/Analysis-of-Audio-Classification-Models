import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from fileUtils import *
from ClassifierStateSaver import ClassifierStateSaver 
import time
import pickle

classifier_pickle_path = 'SGD_Trained_Classifier.bin'

if __name__ == '__main__':
    path = './nsynth-transformed2/nsynth-transformed2/nsynth-train-tranformed2/audio'
    allFileDict = getFileDict(path)

    allClasses = list(allFileDict.keys())

    fileDict = {}
    classes = allClasses[:]

    # generate classifier encoding
    le = preprocessing.LabelEncoder()
    le.fit(classes)

    path = './nsynth-transformed2/nsynth-transformed2/nsynth-test-tranformed2/audio'
    unfilteredTestDict = getFileDict(path)
    testDict = {}
    for cl in classes:
        if cl in unfilteredTestDict:
            testDict[cl] = unfilteredTestDict[cl]

    #create classifier
    with open(classifier_pickle_path, 'rb') as fp:
        clf = pickle.load(fp).clf  

    testTupleList = []
    for (k,v) in testDict.items():
        testTupleList+=[(k,file) for file in v]
    print("Loading Test Set")
    testLabels , testData = loadFiles(testTupleList, path)
    # X,Y inputs for sklearn
    print("Scaling Test Set")
    X = preprocessing.scale(testData)
    Y = le.transform(testLabels)

    prediction = clf.predict(X)

    report = classification_report(Y,prediction,target_names = classes)
    print(report)

    accuracy = accuracy_score(Y,prediction)
    print(f"classifier accuracy: {accuracy}")
