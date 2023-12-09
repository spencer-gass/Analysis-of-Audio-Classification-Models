from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from datetime import datetime
from keras.models import model_from_json
from keras.utils import plot_model

class modelHandler():
    def __init__(self):
        return

    def makeVGGTypeAModel(self, num_categories=2, input_shape=(256, 256, 1)):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='sigmoid', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(4, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='sigmoid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), activation='sigmoid'))
        model.add(Conv2D(256, (3, 3), activation='sigmoid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), activation='sigmoid'))
        model.add(Conv2D(512, (3, 3), activation='sigmoid'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), activation='sigmoid'))
        model.add(Conv2D(512, (3, 3), activation='sigmoid'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(num_categories, activation='softmax'))
        print('built new model')
        return model

    def make1DVGGTypeAModel(self, num_categories=2, input_shape=(256, 1), kernel_size=3):
        model = Sequential()

        model.add(Conv1D(64, kernel_size, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=4))
        #model.add(Dropout(0.25))

        model.add(Conv1D(128, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        #model.add(Dropout(0.25))

        model.add(Conv1D(256, kernel_size, activation='relu'))
        model.add(Conv1D(256, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        #model.add(Dropout(0.25))

        model.add(Conv1D(512, kernel_size, activation='relu'))
        model.add(Conv1D(512, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        #model.add(Dropout(0.25))

        model.add(Conv1D(512, kernel_size, activation='relu'))
        model.add(Conv1D(512, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2048, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_categories, activation='softmax'))
        print('built new model')
        self._print_io(model)
        return model

    def makeMiniVGGnetModel(self, num_categories=2, input_shape=(256,256,1), dropout_rate=0.25, dense_width=256, kernel_shape=(3, 3)):
        # build a new sequential model too form a mini-VGGnet style classifier
        model = Sequential()
        # 32 convolution filters of size 3x3 each.
        # input image shape (num_images,256,256,3)
        model.add(Conv2D(32, kernel_shape, activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, kernel_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(dropout_rate))

        model.add(Conv2D(64, kernel_shape, activation='relu'))
        model.add(Conv2D(64, kernel_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(dense_width, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_categories, activation='softmax'))
        print('built new model')
        return model

    def make1DMiniVGGnetModel(self, num_categories=2, input_shape=(256,1), dropout_rate=0.1, dense_width=256, ):
        # build a new sequential model too form a mini-VGGnet style classifier
        model = Sequential()
        # 32 convolution filters of size 3x3 each.
        # input image shape (num_images,256,256,3)
        model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(dense_width, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(num_categories, activation='softmax'))
        print('built new model')
        #self._print_io(model)
        return model

    def makeMLPModel(self, num_categories=2, input_dim=2048, num_hidden_layers=1, dropout_rate=0.5, hidden_dim=2048):
        model = Sequential()

        model.add(Dense(hidden_dim, activation='tanh', input_shape=(input_dim,)))
        model.add(Dropout(dropout_rate))

        for i in range(num_hidden_layers):
            model.add(Dense(hidden_dim, activation='relu'))
            model.add(Dropout(dropout_rate))

        model.add(Dense(num_categories, activation='softmax'))
        self.plotModel(model,'MLP_diag')
        #for layer in model.layers:
        #    print(layer.input_shape, layer.output_shape, layer.get_config()['name'], layer.output)
        return model

    def loadModel(self, model_json_path, model_weights_path):
        # load model
        json_file = open(model_json_path)
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights_path)
        print('loaded model ' + model_json_path + ' from disk')
        return model

    def plotModel(self, model, fname):
        plot_model(model, to_file=fname)

    def saveModel(self, model, path):
        # save the model
        timestamp = str(datetime.today()).replace('-', '').replace(' ', '_').replace(':', '').split('.')[0][4:-2]
        model_json = model.to_json()

        # save model with time stamp
        with open(path + '/model_' + timestamp + '.json', 'w') as j:
            j.write(model_json)
            j.close()
        model.save_weights(path + '/weights_' + timestamp + '.h5')

        # save model as latest
        with open(path + '/latest_model.json', 'w') as j:
            j.write(model_json)
            j.close()
        model.save_weights(path + '/latest_weights.h5')

        print('saved model to disk')

    def _print_io(self, model):
        for l in model.layers:
            c = l.get_config()
            print(c['name'] + ' ' + str(l.input_shape) + ' ' + str(l.output_shape))