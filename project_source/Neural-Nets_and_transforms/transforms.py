from librosa.core import stft
from librosa.feature import mfcc
from librosa.feature import *
import numpy as np
from constants import *

'''
MFCC vs SPECTROGRAM
    mfcc        large (high resolution on the first second) xdim ~ 32000
                small (large spacing on the whole 4 seocnds and smaller fft) xdim ~ 4096
    spectrogram large
                small
                
OTHERS
    centroid
    bandwidth
    flatness
    zero crossing rate

FULL VECTOR
    mfcc + others
    spectrogram + others

CLIPPED TIME


NOTES:
    GPU cant handle an input dimension of more than 32K 
    "OTHERS" are one dimensional so they can be higher resolution
     
    

    mfcc_small 
    mfcc_large 
    stft_small 
    stft_large       
                  50000 points,     10000 points      10000 points
                  10 epochs         5 epochs          5 spochs
    mfcc_small    0.125480153649    0.0961538461538   0.125            0.105769230769
    mfcc_large    0.10371318822     0.259615384615    0.105769230769   0.253205128205
    stft_small    0.10371318822     0.121794871795    0.0897435897436  0.070512820512
    stft_large    0.0537772087068   0.0705128205128   0.0608974358974  0.060897435897    
    centroid                        0.099358974359    0.240384615385   0.189102564103
    bandwidth                       0.189102564103    0.0448717948718  0.253205128205   
    slope                           0.259615384615    0.221153846154   0.262820512821
    zero_crossing                   0.259615384615    0.240384615385   0.253205128205
    time domain                     0.084             0.038            0.134
    
    feature_vector3     0.114   0.209    0.175
    feature_vector2     0.178   0.209    0.105    0.053  
    feature_vector1                               0.166
                                                  

'''

def mfcc_large(x):
    x = x.astype(np.float32, copy=False)
    return mfcc(y=x[:16000],
                sr=FS,
                n_mfcc=512,
                hop_length=256,
                n_fft = 512)


def mfcc_small(x):
    x = x.astype(np.float32, copy=False)
    return mfcc(y=x,
                sr=FS,
                n_mfcc=128,
                hop_length=2048,
                n_fft = 128)


def stft_large(x):
    x = x.astype(np.float32, copy=False)
    return stft(y=x[:16000],
                n_fft=1024,
                hop_length=256)


def stft_small(x):
    x = x.astype(np.float32, copy=False)
    return stft(y=x,
                n_fft=256,
                hop_length=2048)


def sp_centroid(x):
    x = x.astype(np.float32, copy=False)
    return spectral_centroid(y=x, sr=FS, n_fft=2048, hop_length=512)


def sp_bandwidth(x):
    x = x.astype(np.float32, copy=False)
    return spectral_bandwidth(y=x, sr=FS, n_fft=2048, hop_length=512)


def sp_slope(x):
    x = x.astype(np.float32, copy=False)
    return spectral_flatness(y=x, n_fft=2048, hop_length=512)


def zcr(x):
    x = x.astype(np.float32, copy=False)
    return zero_crossing_rate(y=x, frame_length=2048, hop_length=512)

def td(x):
    x = x.astype(np.float32, copy=False)
    return x[:16000]

def feature_vector1(x):
    x = x.astype(np.float32, copy=False)
    hl = 128
    hl4 = hl * 4
    m = mfcc(y=x[:16000],
             sr=FS,
             n_mfcc=256,
             hop_length=hl,
             n_fft = 256)
    c = spectral_centroid(y=x, sr=FS, n_fft=2048, hop_length=hl4)
    b = spectral_bandwidth(y=x, sr=FS, n_fft=2048, hop_length=hl4)
    s = spectral_flatness(y=x, n_fft=2048, hop_length=hl4)
    z = zero_crossing_rate(y=x, frame_length=2048, hop_length=hl4)

    r = np.concatenate((m, c, b, s, z))
    return r

def feature_vector2(x):
    x = x.astype(np.float32, copy=False)
    hl = 1024
    m = mfcc(y=x,
             sr=FS,
             #n_mfcc=256,
             hop_length=hl,
             n_fft = 256)

    c = spectral_centroid(y=x, sr=FS, n_fft=256, hop_length=hl)
    b = spectral_bandwidth(y=x, sr=FS, n_fft=256, hop_length=hl)
    s = spectral_flatness(y=x, n_fft=256, hop_length=hl)
    z = zero_crossing_rate(y=x, frame_length=256, hop_length=hl)

    r = np.concatenate((m, c, b, s, z))
    return r

transforms = {'mfcc_large': mfcc_large,
              'mfcc_small': mfcc_small,
              'stft_large': stft_large,
              'stft_small': stft_small,
              'centroid':      sp_centroid,
              'bandwidth':     sp_bandwidth,
              'slope':         sp_slope,
              'zero_crossing': zcr,
              'time_domain':   td,
              'feature_vector1': feature_vector1
              }