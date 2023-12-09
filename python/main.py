'''

7 days

architect a NN
    full training on MLP
        normalized / over sampled data / sequenc bug fix

architect a CNN
    miniVGG
        sequence bug fix
    validation runs on
        VGG-A
            any result...
        VGG-E
        Inception

'''

import numpy as np
from time import time
from constants import *
from modelHandler import modelHandler
from modelTrainer import modelTrainer
from dataLoader2 import *


def main():

    d = CNN_DataLoader()
    t = modelTrainer()
    h = modelHandler()

    batch_size = 110

    train_data, validation_data, test_data = d.getTrainAndValidSeq(n=None, batch_size=batch_size)
    print('num samples: ', len(train_data) * batch_size)
    for i in range(0):
        print(train_data[i][1])

    xdim = train_data.getInputDim()
    print('xdim ', xdim)

    model_dir = 'miniVGG'
    model = h.loadModel('/home/sgass/Projects/ECE-6254/semester_project/models/' + model_dir + '/latest_model.json',
                        '/home/sgass/Projects/ECE-6254/semester_project/models/' + model_dir + '/latest_weights.h5')

    # model = h.loadModel('/home/sgass/Projects/ECE-6254/semester_project/models/miniVGG/model_0419_1559.json',
    #                     '/home/sgass/Projects/ECE-6254/semester_project/models/miniVGG/weights_0419_1559.h5')

    # model = h.makeMLPModel(num_categories=11,
    #                        input_dim=xdim,
    #                        num_hidden_layers=2,
    #                        dropout_rate=0.2,
    #                        hidden_dim=8)

    # model = h.makeMiniVGGnetModel(num_categories=11,
    #                               input_shape=xdim,
    #                               dropout_rate=0.0,
    #                               dense_width=256,
    #                               kernel_shape=(3, 3))

    # model = h.makeVGGTypeAModel(num_categories=11,
    #                             input_shape=xdim)

    # model = h.make1DVGGTypeAModel(num_categories=11,
    #                               input_shape=xdim,
    #                               kernel_size=3)

    # model = h.make1DMiniVGGnetModel(num_categories=11,
    #                                 input_shape=xdim,
    #                                 dropout_rate=0.0,
    #                                 dense_width=256)

    t.compileModel(model=model,
                   lr=1e-3,
                   decay=0.0)

    # t.trainModel(model=model,
    #              data_gen=train_data,
    #              batch_size=batch_size,
    #              epochs=10)

    # h.saveModel(model, '/home/sgass/Projects/ECE-6254/semester_project/models/' + model_dir + '/')
    # t.evaluateModel(model, validation_data)
    t.evaluateModel(model, test_data)
    # t.modelPredict(model, validation_data)

start = time()
main()
print('elaped time ', (time() - start) / 60)