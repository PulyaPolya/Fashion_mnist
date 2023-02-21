from __future__ import print_function
from evolution import Evolution
import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
import functions as f
from operator import itemgetter
import time
from tensorflow.keras import layers

import sys
import threading
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread
try:
    range, _print = xrange, print
    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        _print(*args, **kwargs)
        if flush:
            kwargs.get('file', sys.stdout).flush()
except NameError:
    pass
def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt
def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer
class Epoch_Tracker:
  def __init__(self):
     self.epoch = 0
     self.change = True
  def increase(self):
    self.epoch +=1
    self.change = True

epoch_track = Epoch_Tracker()
def random_invert_img(x):
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

start = time.time()
tf.random.set_seed(1234)
# class Logger(object):
#     def __init__(self):
#         self.terminal = sys.stdout
#         self.log = open("log.dat", "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# sys.stdout = Logger()
def define_model(conv1, conv2,conv3, kernel1, kernel2, kernel3, dropout1, dropout2):
    model = tf.keras.models.Sequential([
        RandomInvert(),
        tf.keras.layers.Conv2D(conv1, (kernel1, kernel1), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(conv2, ( kernel2, kernel2), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout1 / 10),
        tf.keras.layers.Conv2D(conv3, (kernel3, kernel3), padding='same', activation='relu'),
        tf.keras.layers.Dropout(dropout2 / 10),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')])
    return model

def print_model(conv1, conv2,conv3, kernel1, kernel2, kernel3, dropout1, dropout2,  l_rate, opt):
    print(f'conv1 =  {conv1} '
          f'\n conv2 =  { conv2}'
          f'\n conv3 =  {conv3}'
          f'\n kernel1 =  {kernel1}'
          f'\n kernel2 =  {kernel2}'
          f'\n kernel3 =  {kernel3}'
          f'\n dropout1 =  {dropout1 / 10}'
          f'\n dropout2 =  {dropout2 / 10} '
          f'\n l_rate =  {l_rate} '
          f'\n opt =  {opt} '
    )


def train_models(evolution,models, numb_iteration, first_run= False, prev_two_val_acc = None):
    val_acc_arr = []
    for hyper_params in models:
        print('-----------------------------------------------')
        print(f'Training for a model number {models.index(hyper_params)+1}, {numb_iteration} iteration')
        print(f'The parameters are: ')
        print_model(conv1=hyper_params[0], conv2=hyper_params[1], conv3=hyper_params[2],  kernel1 = hyper_params[3],
                    kernel2=hyper_params[4], kernel3 = hyper_params[5],
                             dropout1=hyper_params[6], dropout2=hyper_params[7], l_rate=hyper_params[8], opt= hyper_params[9])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.7)
        model = define_model(conv1=hyper_params[0], conv2=hyper_params[1], conv3=hyper_params[2],
                             kernel1=hyper_params[3],kernel2=hyper_params[4], kernel3=hyper_params[5],
                             dropout1=hyper_params[6], dropout2=hyper_params[7])
        if hyper_params[9] == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=hyper_params[8]/10000)
        elif hyper_params[9] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=hyper_params[8] / 10000)
        elif hyper_params[9] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params[8] / 10000)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], run_eagerly=True)
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            callbacks = [early_stop],
                            verbose=1
                            )
        evolution.numb_of_trained_models += 1
        val_acc = max(history.history['val_acc'])
        val_acc_arr.append(val_acc)
    return val_acc_arr

def get_best_model(val_acc_arr, models):
    index_best_model = val_acc_arr.index(max(val_acc_arr))
    best_model = models[index_best_model].copy()
    return best_model


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
input_shape = (28, 28, 1)
x_train, y_train, x_test, y_test = f.edit_data(x_train[:100], y_train[:100],
                                               x_test, y_test)

batch_size = 64
num_classes = 10
epochs = 1
numb_of_runs = 2
@exit_after(28800)
def run_evo():

    best_acc = -1
    evolution = Evolution(numb_of_indiv=4)
    models = evolution.initialize()
    val_acc_arr = train_models(evolution,evolution.individuals, 1, first_run= True)

    full_models = models[:]
    print(val_acc_arr)
    for i in range(numb_of_runs):
        if max(val_acc_arr) > best_acc:
            best_acc = max(val_acc_arr)
            best_model = get_best_model(val_acc_arr, full_models)
            best_iteration = i
        print('---------------------------------------------------')
        print(f'Best model so far:')
        print_model(best_model[0], best_model[1], best_model[2], best_model[3], best_model[4],best_model[5], best_model[6],
                    best_model[7], best_model[8], best_model[9])
        print(f'best_val_acc = {round(best_acc, 4)}')
        print(f'best model accomplished on iteration number {best_iteration}')
        print(evolution.numb_of_trained_models)
        print('---------------------------------------------------')
        print(f'training for {i+2}th time')
        models, full_models= evolution.run_evolution(val_acc_arr)
        indexes = evolution.choose_n_val(val_acc_arr)
        two_best_val = list(itemgetter(*indexes)(val_acc_arr))
        val_acc_arr_temp = train_models(evolution, models, i+2, first_run= False, prev_two_val_acc=sorted(val_acc_arr[-2:]))
        val_acc_arr = two_best_val + val_acc_arr_temp
        print(val_acc_arr)
        print(full_models)
        print(models)
        end = time.time()
        print(f'Total elapsed time {end-start}\n')
        print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end-start)))

    print(f'Best model so far:')
    print_model(best_model[0], best_model[1], best_model[2], best_model[3], best_model[4],best_model[5], best_model[6],
                    best_model[7], best_model[8], best_model[9])
    print(f'best_val_acc = {round(best_acc, 4)}')
    f.save_evolution_results(number_of_models = evolution.numb_of_trained_models, conv1=best_model[0], conv2=best_model[1], conv3=best_model[2], lr=best_model[9],
                             kernel1=best_model[3], kernel2=best_model[4], kernel3=best_model[5], opt=best_model[8],
                             dropout1=best_model[6],dropout2=best_model[7], val_acc=round(best_acc, 4), number=2)
    # # f = open("demofile2.txt", "a")
    # # f.write(f'\n Best model so far: conv1 =  {best_model[0]} \n conv2 =  { best_model[1]} '
    # #         f'\n conv3 =  {best_model[2]}'
    # #         f'\n kernel1 =  {best_model[3]}'
    # #         f'\n kernel2 =  {best_model[4]}'
    # #       f'\n kernel3 =  {best_model[5]}'
    # #       f'\n dropout1 =  {best_model[6] / 10}'
    # #       f'\n dropout2 =  {best_model[7] / 10} '
    # #       f'\n l_rate =  { best_model[9]} '
    # #         f'\n opt =  {best_model[8]} ')
    # # f.write(f'\n best_val_acc = {round(best_acc, 4)}\n')
    # # f.write(f'****************************************** \n')
    # # f.close()
f.save_evolution_results(number_of_models = '' ,conv1='40-140', conv2='40-100', conv3='32-80', lr='5--15',
                         kernel1='3--7', kernel2='3--9', kernel3='3--15', opt='',
                         dropout1='3--6',dropout2='3--6', val_acc='', number=0)
run_evo()
