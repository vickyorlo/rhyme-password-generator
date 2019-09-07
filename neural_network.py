from keras.layers import Conv1D, UpSampling1D, Input, MaxPooling1D, BatchNormalization, Dense, Flatten,Dropout
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adamax
from keras.utils import to_categorical
from math import ceil
import numpy
import keras
import os


class NeuralNetwork(object):
    def __init__(self, epochs=10, batch_size=1, path_to_model=None):

        self.epochs = epochs
        self.batch_size = batch_size
        with open("x_batch_16.txt","r") as x, open("y_batch_16.txt",'r') as y:
            self.x_batch = [ [ [ord(ch)-96] for ch in line.rstrip() ] for line in x ]
            self.y_batch = [ [ [ord(ch)-96] for ch in line.rstrip() ] for line in y ]
            self.training_set_size = len(self.x_batch)

        self.x_batch = to_categorical(numpy.array(self.x_batch))
        self.y_batch = to_categorical(numpy.array(self.y_batch))

        #print(self.x_batch[0])
        #print(self.x_batch[1])

        #print(self.y_batch[0])
        #print(self.y_batch[1])

        if path_to_model is None:
            self.neural_network_structure()
        else:
            self.model = load_model(path_to_model)

    def neural_network_structure(self):

        self.model = Sequential()

        #encoder - begin

        self.model.add(Conv1D(filters=32,kernel_size=3,activation='relu', padding='same',input_shape=(16,27)))
        self.model.add(Conv1D(filters=32,kernel_size=3,activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=32,kernel_size=3,activation='relu', padding='same'))

        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(filters=64,kernel_size=3,activation='relu', padding='same'))
        self.model.add(Conv1D(filters=64,kernel_size=3,activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=64,kernel_size=3,activation='relu', padding='same'))
       
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(filters=128,kernel_size=3,activation='relu', padding='same'))
        self.model.add(Conv1D(filters=128,kernel_size=3,activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=128,kernel_size=3,activation='relu', padding='same'))

        self.model.add(BatchNormalization())

        #encoder- end

        #decoder - begin

        self.model.add(Conv1D(filters=128,kernel_size=3,activation='relu', padding='same'))
        self.model.add(Conv1D(filters=128,kernel_size=3,activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=128,kernel_size=3,activation='relu', padding='same'))

        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(filters=64,kernel_size=3,activation='relu', padding='same'))
        self.model.add(Conv1D(filters=64,kernel_size=3,activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=64,kernel_size=3,activation='relu', padding='same'))

        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(filters=32,kernel_size=3,activation='relu', padding='same'))
        self.model.add(Conv1D(filters=32,kernel_size=3,activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=32,kernel_size=3,activation='relu', padding='same'))

        #decoder - end 
        
        self.model.add(Dense(27,activation='softmax'))

        self.model.compile(optimizer=Adamax(lr=0.001), loss='categorical_crossentropy', metrics=['mae', 'acc'])
 

    def train(self):
        patience = 10
        tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                                  write_grads=False, write_images=False, embeddings_freq=0,
                                                  embeddings_layer_names=None, embeddings_metadata=None)
        model_names = 'model.{epoch:02d}-{loss:.10f}.hdf5'
        model_checkpoint = ModelCheckpoint(os.path.join('models', model_names), monitor='loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping('loss', patience=patience)
        reduce_lr = ReduceLROnPlateau('loss', factor=0.1, patience=int(patience / 2), verbose=1)


        self.model.fit(self.x_batch,self.y_batch, batch_size=self.batch_size, epochs=self.epochs,callbacks=[tb_callback, early_stop, reduce_lr, model_checkpoint])

    def save_model(self):
        self.model.save_weights('weights_{}e.h5'.format(self.epochs))
        self.model.save('model_{}e_m.h5'.format(self.epochs))

    def run(self):
        self.train()
        self.save_model()