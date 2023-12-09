import json
import wave
from scipy.io import wavfile
from scipy.signal import stft
from librosa.feature import mfcc
from librosa.feature import spectral_centroid
from librosa.feature import spectral_bandwidth
from librosa.feature import spectral_flatness
import numpy as np
from time import time
from constants import *
from dataLoaderWithTransforms import *
import matplotlib.pyplot as plt



def example():
    ##############
    # load labels
    with open(TRAIN_PATH + 'examples.json') as j:
        y = json.load(j)
        # y is a 2D dictionary -> d[sample_name][feature_name]

    print(y['bass_acoustic_000-026-050'])

    ##################
    # load audio data
    #with wave.open(TRAIN_PATH + 'audio/bass_acoustic_000-026-050.wav', 'r') as f:
        #print(f.getparams())
        # nchannels=1 (ie mono)
        # sampwidth=2 (bytes)
        # framerate=16000 (samples/second)
        # nframes=64000
        # comptype='NONE'

    fname = 'audio/keyboard_synthetic_001-055-127.wav'
    framerate, x0 = wavfile.read(TRAIN_PATH + fname)
    print('x0 shape', x0.shape)
    x0 = x0.astype(np.float32, copy=False)

    #############
    # transforms
    #m = mfcc(x0, framerate, n_mfcc=256, hop_length=64, n_fft=256)
    #m = mfcc(x0,framerate,)
    m = spectral_centroid(x0,framerate)
    #m = spectral_flatness(x0)
    n = spectral_bandwidth(x0,framerate)
    #f,t,m = stft(x0)
    #m = np.abs(m)

    print(m.shape)
    print(m)

    # write it to a file
    ##np.save(file=TRANSFORM_PATH + fname, arr=m, allow_pickle=False)

    # read it from the file
    ##m = np.load(file=TRANSFORM_PATH + fname + '.npy')

    #print(m.shape)
    #print(m)

    plt.plot(m[0])
    plt.plot(n[0])
    #plt.imshow(m, interpolation='nearest')
    plt.show()
#example()

################## Stuff to write the data set to a file############################


import numpy as np
from time import time
from constants import *
from modelHandler import modelHandler
from modelTrainer import modelTrainer
from dataLoaderWithTransforms import *
from transforms import *


def train_a_model(paths, transform):
    h = modelHandler()
    t = modelTrainer()

    train_data, holdout_data = getTrainSubsetSeq(paths, names, transform)
    xdim = train_data.getInputDim()
    print(xdim)

    model = h.makeMLPModel(num_categories=11,
                           input_dim=xdim,
                           num_hidden_layers=5,
                           dropout_rate=0.3)
    t.compileModel(model)
    t.trainModel(model=model,
                 data_gen=train_data,
                 batch_size=20,
                 epochs=5)

    acc = t.evaluateModel(model, holdout_data)

    return acc

def compare_transforms():
    start = time()

    paths, names = getPaths(20000)

    results = dict()

    for t in transforms:
        name = t
        print(name)
        t = transforms[t]
        acc = train_a_model(paths, t)
        results[name] = acc

    for name in results:
        print(name, results[name])

    print('elaped time ', (time()-start)/60)

def transform_and_write_to_file(src, dest, func):
    # src: source path to the .wav files
    # dest: destination path to store tranformed data
    # func: function that transforms the data, must return an numpy array

    # get file names
    paths = glob(src + 'audio/*.wav')
    num_paths = len(paths)
    print('num file names: ', num_paths)

    count = 0

    start = time()

    for path in paths:

        # read the file
        fs, x = wavfile.read(path)

        # transform the data
        X = func(x)

        # write it to a file
        fname = path.split('/')[-1]
        new_path = dest + 'audio/' + fname
        #X = X.astype(dtype=np.float16, copy=False)
        np.save(new_path, arr=X, allow_pickle=False)

        count += 1
        if count%1000 == 0:
            t_diff = time() - start
            eta = (num_paths-count) * (t_diff/count)
            print (eta/60, 'minutes left')

transform_and_write_to_file(src=RAW_TRAIN_PATH,
                            dest='/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed2/nsynth-train-tranformed2/',
                            func=feature_vector2)