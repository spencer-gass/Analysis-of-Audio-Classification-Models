from DataSequence import *
import numpy as np
from constants import *
from glob import glob
from random import shuffle
import json

class dataLoader():
    def __init__(self):
        pass

    def getTrainAndValidSeq(self, n, batch_size):
        raise NotImplementedError

    def _getPathsAndLabels(self, path, glob_ext, n):
        paths, names = self._getPaths(path + glob_ext, n)
        labels = self._getLabels(path, names)
        ordered_paths = self._getOrderedPaths(paths, labels)
        return ordered_paths, labels

    def _getOrderedPaths(self, paths, labels):
        op = dict()
        for i in range(len(paths)):
            label = labels[i]
            path = paths[i]
            if label not in op:
                op[label] = [path]
            else:
                op[label].append(path)
        return op

    def _getPaths(self, glob_path, num_paths=None):
        paths = glob(glob_path)
        shuffle(paths)
        names = [p.split('/')[-1].split('.')[0] for p in paths]
        length = len(names)
        if num_paths is None or num_paths >= length:
            return paths, names
        else:
            return paths[:num_paths], names[:num_paths]

    def _getLabels(self, path, names):
        num_paths = len(names)
        with open(path + 'examples.json') as j:
            y = json.load(j)
            # y is a 2D dictionary -> d[sample_name][feature_name]

        labels = [y[names[i]]['instrument_family_str'] for i in range(num_paths)]

        return labels


class NN_DataLoader(dataLoader):
    def __init__(self):
        dataLoader.__init__(self)

    def getTrainAndValidSeq(self, n, batch_size):
        x_train_paths, y_train = self._getPathsAndLabels(TRAIN_PATH, 'audio/*.wav.npy', n)
        x_valid_paths, y_valid = self._getPathsAndLabels(VALIDATION_PATH, 'audio/*.wav.npy', n)
        x_test_paths, y_test = self._getPathsAndLabels(TEST_PATH, 'audio/*.wav.npy', n)

        return NN_DataSequence(x_train_paths, y_train, batch_size), \
               NN_DataSequence(x_valid_paths, y_valid, batch_size), \
               NN_DataSequence(x_test_paths, y_test, batch_size)



class NN_RawDataLoader(dataLoader):
    def __init__(self):
        dataLoader.__init__(self)

    def getTrainAndValidSeq(self, n, batch_size):
        x_train_paths, y_train = self._getPathsAndLabels(RAW_TRAIN_PATH, 'audio/*.wav', n)
        x_valid_paths, y_valid = self._getPathsAndLabels(RAW_VALIDATION_PATH, 'audio/*.wav', n)
        x_test_paths, y_test = self._getPathsAndLabels(RAW_TEST_PATH, 'audio/*.wav', n)

        return NN_RawDataSequence(x_train_paths, y_train, batch_size), \
               NN_RawDataSequence(x_valid_paths, y_valid, batch_size), \
               NN_RawDataSequence(x_test_paths, y_test, batch_size)


class CNN_DataLoader(dataLoader):
    def __init__(self):
        dataLoader.__init__(self)

    def getTrainAndValidSeq(self, n, batch_size):
        x_train_paths, y_train = self._getPathsAndLabels(TRAIN_PATH, 'audio/*.wav.npy', n)
        x_valid_paths, y_valid = self._getPathsAndLabels(VALIDATION_PATH, 'audio/*.wav.npy', n)
        x_test_paths, y_test = self._getPathsAndLabels(TEST_PATH, 'audio/*.wav.npy', n)

        return CNN_DataSequence(x_train_paths, y_train, batch_size), \
               CNN_DataSequence(x_valid_paths, y_valid, batch_size), \
               CNN_DataSequence(x_test_paths, y_test, batch_size)

class CNN_RawDataLoader(dataLoader):
    def __init__(self):
        dataLoader.__init__(self)

    def getTrainAndValidSeq(self, n, batch_size):
        x_train_paths, y_train = self._getPathsAndLabels(RAW_TRAIN_PATH, 'audio/*.wav', n)
        x_valid_paths, y_valid = self._getPathsAndLabels(RAW_VALIDATION_PATH, 'audio/*.wav', n)
        x_test_paths, y_test = self._getPathsAndLabels(RAW_TEST_PATH, 'audio/*.wav', n)

        return CNN_RawDataSequence(x_train_paths, y_train, batch_size), \
               CNN_RawDataSequence(x_valid_paths, y_valid, batch_size), \
               CNN_RawDataSequence(x_test_paths, y_test, batch_size)

