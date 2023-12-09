from keras.utils import Sequence
from constants import *
from scipy.io import wavfile
import numpy as np

class DataSequence(Sequence):

    def __init__(self, x_paths, y_labels, batch_size):
        # x paths is a dictionary of x[label] = list(paths)
        self.x_paths = x_paths
        #self.y = y_labels
        self.batch_size = batch_size
        self.label_names = list(self.x_paths.keys())
        self.num_labels = len(self.label_names)
        self.num_points = len(y_labels)
        self.num_points_per_label = [len(self.x_paths[self.label_names[i]]) for i in range(self.num_labels)]
        self.length = max(self.num_points_per_label) * self.num_labels

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))

    def __getitem__(self, idx):
        idx *= self.batch_size
        x_list = list()
        y_list = list()
        for i in range(self.batch_size):
            x ,y = self._get_single_item(idx+i)
            x_list.append(x)
            y_list.append(y)
        return np.array(x_list), np.array(y_list)

    def _get_single_item(self, idx):
        label_idx = idx % self.num_labels
        label = self.label_names[label_idx]
        i = (idx // self.num_labels) % self.num_points_per_label[label_idx]
        path = self.x_paths[label][i]

        xi = self.openFile(path)
        yi = name2vector[label]
        return xi, yi

    def on_epoch_end(self):
        pass

    def openFile(self, path):
        raise NotImplementedError

    def serialize(self, x):
        s = x.shape
        p = 1
        for d in s:
            p *= d
        return x.reshape((p), order='F')

    def reshapeData(self,x):
        s = list(x.shape)
        s = s + [1]
        s = tuple(s)
        return x.reshape(s)

    def getNumSamples(self):
        return self.num_points

    def getInputDim(self):
        x = self.__getitem__(0)
        x = len(x[0][0])
        return x


class NN_DataSequence(DataSequence):

    def __init__(self, x_paths, y_labels, batch_size):
        DataSequence.__init__(self, x_paths, y_labels, batch_size)

    def openFile(self, path):
        x = np.load(file=path)
        m = np.max(x)
        x = x/m
        return self.serialize(x)



class NN_RawDataSequence(DataSequence):

    def __init__(self, x_paths, y_labels, batch_size):
        DataSequence.__init__(self, x_paths, y_labels, batch_size)
        self.num_sampels = 64000

    # everything is the same as in the neural net data
    # generator except that you don't serialize the array
    def openFile(self, path):
        fs, x = wavfile.read(path)
        x = x[:self.num_sampels]
        x = x.reshape((1, self.num_sampels))
        m = np.max(x)
        x = x/m
        return x


class CNN_DataSequence(DataSequence):

    def __init__(self, x_paths, y_labels, batch_size):
        DataSequence.__init__(self, x_paths, y_labels, batch_size)

    # everything is the same as in the neural net data
    # generator except that you don't serialize the array
    def openFile(self, path):
        x = np.load(file=path)
        x = self.reshapeData(x)
        m = np.max(x)
        x = x/m
        return x

    def getInputDim(self):
        x = self.__getitem__(0)
        x = np.array(x[0]).shape
        return x[1:]


class CNN_RawDataSequence(DataSequence):

    def __init__(self, x_paths, y_labels, batch_size):
        DataSequence.__init__(self, x_paths, y_labels, batch_size)
        self.num_sampels = 32000

    # everything is the same as in the neural net data
    # generator except that you don't serialize the array
    def openFile(self, path):
        fs, x = wavfile.read(path)
        x = x[:self.num_sampels]
        x = self.reshapeData(x)
        #m = np.max(np.abs(x))
        x = x/65536
        return x

    def getInputDim(self):
        x = self.__getitem__(0)
        x = np.array(x[0]).shape
        return x[1:]