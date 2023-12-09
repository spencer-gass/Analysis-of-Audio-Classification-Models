import numpy as np
from time import time
from constants import *
from modelHandler import modelHandler
from modelTrainer import modelTrainer
from dataLoaderWithTransforms import *
from transforms import *

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
