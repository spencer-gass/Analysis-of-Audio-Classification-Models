import wave
from scipy.io import wavfile
import numpy as np
from time import time
from constants import *
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from random import shuffle
import json


class TransformSequence(Sequence):

    def __init__(self, x_paths, y_labels, transform):
        self.x, self.y = x_paths, y_labels

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xi = self.openAndTransform(self.x[idx])
        yi = name2vector[self.y[idx]].reshape((1,11))
        return xi, yi

    def on_epoch_end(self):
        pass

    def openAndTransform(self, path):
        fs, x = wavfile.read(path)
        X = self.transform(x)
        return self.serialize(X)

    def serialize(self, x):
        s = x.shape
        p = 1
        for d in s:
            p *= d
        return x.reshape((1, p), order='F')

    def getNumSamples(self):
        return len(self.y)

    def getInputDim(self):
        x = self.__getitem__(0)
        x = len(x[0][0])
        return x



def getPaths(num_paths=None):
    paths = glob(TRAIN_PATH + 'audio/*.wav')
    shuffle(paths)
    names = [path.split('/')[-1].split('.')[0] for path in paths]
    if num_paths:
        return paths, names
    else:
        return paths[:num_paths], names[:num_paths]

def getTrainSubsetSeq(paths, names, transform):
    num_paths = len(paths)
    with open(TRAIN_PATH + 'examples.json') as j:
        y = json.load(j)
        # y is a 2D dictionary -> d[sample_name][feature_name]

    labels = [y[names[i]]['instrument_family_str'] for i in range(num_paths)]

    return (TransformSequence(paths[:num_paths // 2], labels[:num_paths // 2], transform),
            TransformSequence(paths[num_paths // 2:num_paths], labels[num_paths // 2:num_paths], transform))













