"""
Worker for Examples 1-4
=======================
This class implements a very simple worker used in the firt examples.
"""

import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import ConfigSpace as CS
import functions as f
import numpy as np
import ConfigSpace.hyperparameters as CSH
from tensorflow.keras import layers

from hpbandster.core.worker import Worker

dataset = 'ORACLE'
# dataset = 'FASHION'
num_classes = 10
input_shape = (28, 28, 1)
x_train_orig, y_train_orig,  x_test_orig, y_test_orig = f.choose_dataset(dataset)
# x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
size_data = x_train_orig.shape[0]
batch_size = 64
epochs =10
class Epoch_Tracker:
  def __init__(self):
     self.epoch = 0
     self.change = True
  def increase(self):
    self.epoch +=1
    self.change = True

def random_invert_img(x):
  #print(epoch_track.epoch)
  if epoch_track.epoch >= epochs:
     return x
  x_temp = x.numpy()
  x_temp = x_temp.reshape(x_temp.shape[0], 28,28)
  x_shifted = []
  for image in x_temp:
       x_shifted.append(f.shift_image_np(image))
  x_shifted = np.array(x_shifted)

  x_result = x_shifted.reshape(x_temp.shape[0],28,28,1)

  return x_result
def random_invert():
  return layers.Lambda(lambda x: random_invert_img(x))

random_invert = random_invert()
class RandomInvert(layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, x):
    return random_invert_img(x)

epoch_track = Epoch_Tracker()
tf.random.set_seed(1234)
import logging
val = []

class MyWorker(Worker):
    def __init__(self, x_train, y_train,x_val, y_val, **kwargs):
            super().__init__(**kwargs)

            self.batch_size = 64

            img_rows = 28
            img_cols = 28
            self.num_classes = 10
            self.time = time.time()

            # the data, split between train and test sets
            # (x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()
            # x_train = x_train[:64]
            # y_train = y_train[:64]
            # x_val = x_val[-64:]
            # y_val = y_val[-64:]

            if K.image_data_format() == 'channels_first':
                    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
                    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
                    self.input_shape = (1, img_rows, img_cols)
            else:
                    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
                    self.input_shape = (img_rows, img_cols, 1)


            x_train = x_train.astype('float32')
            x_val = x_val.astype('float32')
            # zero-one normalization
            x_train /= 255
            x_val /= 255


            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_val = keras.utils.to_categorical(y_val, self.num_classes)


            self.x_train, self.y_train = x_train, y_train
            # self.x_validation, self.y_validation = x_train, y_train
            self.x_val, self.y_val   = x_val, y_val

            self.input_shape = (img_rows, img_cols, 1)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        model = Sequential()
        model.add(RandomInvert())
        model.add(Conv2D(config['num_filters_1'], kernel_size=(2*config['num_kernel_1'] +1, 2*config['num_kernel_1'] +1),
                         padding='same',
                         activation='relu',
                         input_shape=self.input_shape))

        model.add(Conv2D(config['num_filters_2'], kernel_size=(2*config['num_kernel_2'] +1, 2*config['num_kernel_2'] +1),
                         padding='same',
                         activation='relu'))
        model.add(keras.layers.MaxPool2D())
        model.add(Dropout(config['dropout_rate1']/10))
        model.add(Conv2D(config['num_filters_3'], kernel_size=(2*config['num_kernel_3'] +1, 2*config['num_kernel_3'] +1),
                         padding='same',
                         activation='relu'))
        model.add(Dropout(config['dropout_rate2']/10))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

        if config['optimizer'] == 'adam':

            optimizer = keras.optimizers.Adam(learning_rate=config['lr']/ 10000)
        elif config['optimizer'] == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=config['lr']/ 10000)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=config['lr']/ 10000)


        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'],run_eagerly=True)

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=int(budget),
                  verbose=1,
                  validation_data=(self.x_val, self.y_val))

        train_score = model.evaluate(self.x_train, self.y_train, verbose=1)
        val_score = model.evaluate(self.x_val, self.y_val, verbose=1)
        # test_score = model.evaluate(self.x_test, self.y_test, verbose=1)
        end = time.time()
        print(f'total elapsed time {time.strftime("%H:%M:%S", time.gmtime(end-self.time))}')
        #print(f'test_acc : {test_score[1]},train accuracy {train_score[1]}, validation accuracy {val_score[1]} ')
        val.append(val_score[1])
        print(f'max valacc: {max(val)}')
        # import IPython; IPython.embed()
        return ({
            'loss': 1 - val_score[1],  # remember: HpBandSter always minimizes!
            'info': {       #'test accuracy': test_score[1],
                     'train accuracy': train_score[1],
                     'validation accuracy': val_score[1],
                     'number of parameters': model.count_params(),
                     'total elapsed time': time.strftime("%H:%M:%S", time.gmtime(end-self.time))
                     }

        })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformIntegerHyperparameter('lr', lower=5, upper=15, log=True)

        optimizer = CSH.CategoricalHyperparameter('optimizer', ['adam', 'nadam', 'RmsProp'])


        cs.add_hyperparameters([lr, optimizer])

        #num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, default_value=2)

        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=40, upper=140, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=40, upper=100, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=32, upper=80,  log=True)
        num_kernel_1 = CSH.UniformIntegerHyperparameter('num_kernel_1', lower=1, upper=3)
        num_kernel_2 = CSH.UniformIntegerHyperparameter('num_kernel_2', lower=1, upper=4)
        num_kernel_3 = CSH.UniformIntegerHyperparameter('num_kernel_3', lower=1, upper=7)

        cs.add_hyperparameters([num_filters_1, num_filters_2, num_filters_3])
        cs.add_hyperparameters([ num_kernel_1,num_kernel_2,num_kernel_3])
        dropout_rate1 = CSH.UniformIntegerHyperparameter('dropout_rate1', lower=3, upper=6, log=True)
        dropout_rate2 = CSH.UniformIntegerHyperparameter('dropout_rate2', lower=3, upper=6, log=True)
        #num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, log=True)

        cs.add_hyperparameters([dropout_rate1,dropout_rate2])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        # cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        # cs.add_condition(cond)

        # You can also use inequality conditions:
        # cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        # cs.add_condition(cond)
        #
        # cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        # cs.add_condition(cond)

        return cs