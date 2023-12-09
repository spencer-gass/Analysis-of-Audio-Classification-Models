from glob import glob
import numpy as np

# This is an example of how to open the .npy files
# unless you have 40GB of ram you probably can't load all of the data at once though
def load_data(path_to_the_data):

    paths = glob(path_to_the_data + '*')
    num_paths = len(paths)
    print(num_paths)

    data = np.zeros(shape=(num_paths, 260, 126))
    # data[:256][:] = mfcc
    # data[256][:] = spectral centroid
    # data[257][:] = spectral bandwidth
    # data[258][:] = spectral slope
    # data[259][:] = zero crossing rate

    for i in range(num_paths):
        data[i] = np.load(file=paths[i])
        break

    return data
