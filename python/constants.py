import numpy as np

############
# constants
TEST_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed/nsynth-test-tranformed1/'
VALIDATION_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed/nsynth-valid-tranformed1/'
TRAIN_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed/nsynth-train-tranformed1/'

TEST_PATH2 = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed2/nsynth-test-tranformed2/'
VALIDATION_PATH2 = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed2/nsynth-valid-tranformed2/'
TRAIN_PATH2 = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-transformed2/nsynth-train-tranformed2/'

TRANSFORM_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-train-tranformed1-tranformed1/'

RAW_TEST_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-test/'
RAW_VALIDATION_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-valid/'
RAW_TRAIN_PATH = '/home/sgass/Projects/ECE-6254/semester_project/nsynth/nsynth-train/'

FS = 16000  # sampling frequency

name2vector = { 'bass'      :np.array([ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'brass'     :np.array([ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'flute'     :np.array([ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'guitar'    :np.array([ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'keyboard'  :np.array([ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'mallet'    :np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'organ'     :np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                'reed'      :np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                'string'    :np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                'synth_lead':np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                'vocal'     :np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
               }

