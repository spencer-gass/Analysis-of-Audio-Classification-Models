from keras.optimizers import SGD

class modelTrainer():
    def __init__(self):
        return

    def compileModel(self, model, lr=1e-3, decay=1e-6):
        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def trainModel(self, model, data_gen, batch_size, epochs):
        model.fit_generator(data_gen,
                            steps_per_epoch=len(data_gen),
                            epochs=epochs,
                            use_multiprocessing=False)

    def evaluateModel(self, model, data_gen):
        batch_size = 10
        score = model.evaluate_generator(data_gen,
                                         steps=len(data_gen) ,
                                         use_multiprocessing=False)
        print("\n%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        return score[1]

    def modelPredict(self, model, data_gen):
        print('makeing predictions')
        r = model.predict_generator(data_gen, steps=len(data_gen))
        print('writing to file')
        with open('predictions', 'w') as f:
            for each in r:
                f.write(str(each))
                f.write('\n')
            f.close()
        print('predictions were written to file')

