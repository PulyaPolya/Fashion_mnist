from __future__ import print_function
from evolution import Evolution
import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
import functions as f
from operator import itemgetter
import time
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard

import sys
import threading
from time import sleep

# dataset = 'ORACLE'
dataset = 'FASHION'
x_train_orig, y_train_orig,  x_test_orig, y_test_orig = f.choose_dataset(dataset)
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


def train_models(evolution,models, numb_iteration, x_train, y_train, x_val, y_val):
    val_acc_arr = []
    for hyper_params in models:
        print('-----------------------------------------------')
        print(f'Training for a model number {models.index(hyper_params)+1}, {numb_iteration} iteration')
        print(f'The parameters are: ')
        print_model(conv1=hyper_params[0], conv2=hyper_params[1], conv3=hyper_params[2],  kernel1 = hyper_params[3],
                    kernel2=hyper_params[4], kernel3 = hyper_params[5],
                             dropout1=hyper_params[6], dropout2=hyper_params[7], l_rate=hyper_params[8], opt= hyper_params[9])

        # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.7)
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

        tensorboard = TensorBoard(log_dir='cross-validation/{}'.format(NAME), update_freq='batch', )
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=30,
                            validation_data = (x_val,y_val),
                            #validation_split=0.1,
                            callbacks = [tensorboard],
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

# x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train,
#                                                x_test, y_test)
# print('please enter the number of fold')
# fold_numb = input()
# if fold_numb == '1':
#     x_val = x_train[:12000]
#     x_train = x_train[12000:]
#     y_val = y_train[:12000]
#     y_train = y_train[12000:]
#     NAME = "Evolution_fold1"
# elif fold_numb == '2':
#     x_val = x_train[12000:24000]
#     x_train = np.concatenate((x_train[:12000],x_train[24000:] ), axis = 0)
#     y_val = y_train[12000:24000]
#     y_train = np.concatenate((y_train[:12000], y_train[24000:]), axis=0)
#     NAME = "Evolution_fold2"
# elif fold_numb == '3':
#     x_val = x_train[24000:36000]
#     x_train = np.concatenate((x_train[:24000], x_train[36000:]), axis=0)
#     y_val = y_train[24000:36000]
#     y_train = np.concatenate((y_train[:24000], y_train[36000:]), axis=0)
#     NAME = "Evolution_fold3"
# elif fold_numb == '4':
#     x_val = x_train[36000:48000]
#     x_train = np.concatenate((x_train[:36000], x_train[48000:]), axis=0)
#     y_val = y_train[36000:48000]
#     y_train = np.concatenate((y_train[:36000], y_train[48000:]), axis=0)
#     NAME = "Evolution_fold4"
# elif fold_numb == '5':
#     x_val = x_train[48000:60000]
#     x_train = x_train[:48000]
#     y_val = y_train[48000:60000]
#     y_train = y_train[:48000]
#     NAME = "Evolution_fold5"

def run_evo(x_train, y_train, x_val, y_val):
    start = time.time()
    best_acc = -1
    evolution = Evolution(numb_of_indiv=4)
    models = evolution.initialize()
    val_acc_arr = train_models(evolution,evolution.individuals, 1, x_train, y_train, x_val, y_val)

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
        #global number
        #number += 1
        # f.save_evolution_results(number_of_models=evolution.numb_of_trained_models, conv1=best_model[0],
        #                          conv2=best_model[1], conv3=best_model[2], lr=best_model[8],
        #                          kernel1=best_model[3], kernel2=best_model[4], kernel3=best_model[5], opt=best_model[9],
        #                          dropout1=best_model[6], dropout2=best_model[7], val_acc=round(best_acc, 4), number=number, fold_numb=fold_numb)
        print(f'best model accomplished on iteration number {best_iteration}')
        print(evolution.numb_of_trained_models)
        print('---------------------------------------------------')
        print(f'training for {i+2}th time')
        models, full_models= evolution.run_evolution(val_acc_arr)
        indexes = evolution.choose_n_val(val_acc_arr)
        two_best_val = list(itemgetter(*indexes)(val_acc_arr))
        val_acc_arr_temp = train_models(evolution, models, i+2, x_train, y_train, x_val, y_val)
        val_acc_arr = two_best_val + val_acc_arr_temp
        print(val_acc_arr)
        print(full_models)
        print(models)
        end = time.time()
        elapsed_time = end-start
        print(f'Total elapsed time {elapsed_time}\n')
        print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end-start)))
        global total_time
        if elapsed_time > total_time:
            break

    print(f'Best model so far:')
    print_model(best_model[0], best_model[1], best_model[2], best_model[3], best_model[4],best_model[5], best_model[6],
                    best_model[7], best_model[8], best_model[9])
    print(f'best_val_acc = {round(best_acc, 4)}')
    global number
    number += 1
    f.save_evolution_results(number_of_models = evolution.numb_of_trained_models, conv1=best_model[0], conv2=best_model[1], conv3=best_model[2], lr=best_model[8],
                             kernel1=best_model[3], kernel2=best_model[4], kernel3=best_model[5], opt=best_model[9],
                             dropout1=best_model[6],dropout2=best_model[7], val_acc=round(best_acc, 4),
                             number=number, fold_numb= fold_numb, time = elapsed_time/3600, file_name = "fashion32_evolution_results.csv")
f.save_evolution_results(number_of_models = '' ,conv1='40-140', conv2='40-100', conv3='32-80', lr='5--15',
                         kernel1='3--7', kernel2='3--9', kernel3='3--15', opt='',
                         dropout1='3--6',dropout2='3--6', val_acc='', number=0,fold_numb=0, time = 0, file_name = "fashion32_evolution_results.csv")
# run_evo()

# for fold_numb in folds_numb:
#     debug = 'debug'
#     if debug == 'debug':
#         epochs = 1
#         numb_of_runs = 2
#
#         x_val = x_train[-100:]
#         y_val = y_train[-100:]
#         x_train = x_train[:100]
#         y_train = y_train[:100]
#         NAME = "Evolution_fold" + fold_numb
#         print(f'\n training for the fold number {fold_numb} \n')
#         run_evo()
#     else:
#         numb_of_runs = 100
#         epochs = 9
# (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = fashion_mnist.load_data()
input_shape = (28, 28, 1)
batch_size = 32
num_classes = 10
numb_of_runs = 100
epochs =15
number = 0
total_time = 5*60*60 - 20*60 # whole time for running each fold
folds_numbers = ['1', '2', '3', '4', '5']
x_train_orig, y_train_orig, x_test_orig, y_test_orig = f.edit_data(x_train_orig, y_train_orig,
                                                       x_test_orig, y_test_orig)
if dataset == 'FASHION':
    for fold_numb in folds_numbers:
        if fold_numb == '1':
            x_val = x_train_orig[:12000]
            x_train = x_train_orig[12000:]
            y_val = y_train_orig[:12000]
            y_train = y_train_orig[12000:]
            NAME = "Evolution_fold1"
            print(f'\n training for the fold number {fold_numb} \n')
        elif fold_numb == '2':
            x_val = x_train_orig[12000:24000]
            x_train = np.concatenate((x_train_orig[:12000],x_train_orig[24000:] ), axis = 0)
            y_val = y_train_orig[12000:24000]
            y_train = np.concatenate((y_train_orig[:12000],y_train_orig[24000:]), axis=0)
            NAME = "Evolution_fold2"
            print(f'\n training for the fold number {fold_numb} \n')
        elif fold_numb == '3':
            x_val = x_train_orig[24000:36000]
            x_train = np.concatenate((x_train_orig[:24000], x_train_orig[36000:]), axis=0)
            y_val = y_train_orig[24000:36000]
            y_train = np.concatenate((y_train_orig[:24000], y_train_orig[36000:]), axis=0)
            NAME = "Evolution_fold3"
            print(f'\n training for the fold number {fold_numb} \n')
        elif fold_numb == '4':
            x_val = x_train_orig[36000:48000]
            x_train = np.concatenate((x_train_orig[:36000], x_train_orig[48000:]), axis=0)
            y_val = y_train_orig[36000:48000]
            y_train = np.concatenate((y_train_orig[:36000], y_train_orig[48000:]), axis=0)
            NAME = "Evolution_fold4"
            print(f'\n training for the fold number {fold_numb} \n')

        elif fold_numb == '5':
            x_val = x_train_orig[48000:60000]
            x_train = x_train_orig[:48000]
            y_val = y_train_orig[48000:60000]
            y_train = y_train_orig[:48000]
            NAME = "Evolution_fold5"
            print(f'\n training for the fold number {fold_numb} \n')
        print(f'x_train shape is {x_train.shape}')
        print(f'y_train shape is {y_train.shape}')
        print(f'x_val shape is {x_val.shape}')
        print(f'y_val shape is {y_val.shape}')
        run_evo(x_train, y_train, x_val, y_val)
elif dataset == 'ORACLE':
    folds_train, folds_labels = f.split_dataset(dataset, x_train_orig, y_train_orig)
    for fold_numb in folds_numbers:
        i = folds_numbers.index(fold_numb)
        # number_of_model = numb_of_models[i]
        if fold_numb == '1':
            x_train = np.concatenate((folds_train[1], folds_train[2], folds_train[3], folds_train[4]))
            y_train = np.concatenate((folds_labels[1],folds_labels[2], folds_labels[3], folds_labels[4]))
            x_val = folds_train[0]
            y_val = folds_labels[0]
            # x_val = x_train_orig[-100:]
            # y_val = y_train_orig[-100:]
            # x_train = x_train_orig[:100]
            # y_train = y_train_orig[:100]

        elif fold_numb == '2':
            x_train = np.concatenate((folds_train[0], folds_train[2], folds_train[3], folds_train[4]))
            y_train = np.concatenate((folds_labels[0],folds_labels[2], folds_labels[3], folds_labels[4]))
            x_val = folds_train[1]
            y_val = folds_labels[1]
        elif fold_numb == '3':
            x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[3], folds_train[4]))
            y_train = np.concatenate((folds_labels[0],folds_labels[1], folds_labels[3], folds_labels[4]))
            x_val = folds_train[2]
            y_val = folds_labels[2]
        elif fold_numb == '4':
            x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[4]))
            y_train = np.concatenate((folds_labels[0],folds_labels[1], folds_labels[2], folds_labels[4]))
            x_val = folds_train[3]
            y_val = folds_labels[3]
        elif fold_numb == '5':
            x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[3]))
            y_train = np.concatenate((folds_labels[0],folds_labels[1], folds_labels[2], folds_labels[3]))
            x_val = folds_train[4]
            y_val = folds_labels[4]
        print(f'\n training for the fold number {fold_numb} \n')
        NAME = "Evo_fold" + fold_numb
        run_evo(x_train, y_train, x_val, y_val)
