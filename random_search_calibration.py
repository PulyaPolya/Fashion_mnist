# import tensorflow as tf
# from keras.datasets import fashion_mnist
# import keras
# import functions as f
# import numpy as np
# from tensorflow.keras import layers
# import multiprocessing
# import time
# import keras_tuner as kt
# from tensorflow import keras
# from keras.callbacks import ModelCheckpoint, TensorBoard
# import sys
# import threading
# from time import sleep
#
# # try:
# #     import thread
# # except ImportError:
# #     import _thread as thread
# # try:
# #     range, _print = xrange, print
# #     def print(*args, **kwargs):
# #         flush = kwargs.pop('flush', False)
# #         _print(*args, **kwargs)
# #         if flush:
# #             kwargs.get('file', sys.stdout).flush()
# # except NameError:
# #     pass
# # def quit_function(fn_name):
# #     # print to stderr, unbuffered in Python 2.
# #     print('{0} took too long'.format(fn_name), file=sys.stderr)
# #     sys.stderr.flush() # Python 3 stderr is likely buffered.
# #     thread.interrupt_main() # raises KeyboardInterrupt
# # def exit_after(s):
# #     '''
# #     use as decorator to exit process if
# #     function takes longer than s seconds
# #     '''
# #     def outer(fn):
# #         def inner(*args, **kwargs):
# #             timer = threading.Timer(s, quit_function, args=[fn.__name__])
# #             timer.start()
# #             try:
# #                 result = fn(*args, **kwargs)
# #             finally:
# #                 timer.cancel()
# #             return result
# #         return inner
# #     return outer
#
# num_classes = 10
# input_shape = (28, 28, 1)
# tf.random.set_seed(1234)
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# # x_train, y_train, x_test, y_test = f.edit_data(x_train[:100], y_train[:100], x_test, y_test)
# size_data = x_train.shape[0]
# batch_size = 64
# epochs =9
# class Epoch_Tracker:
#   def __init__(self):
#      self.epoch = 0
#      self.change = True
#   def increase(self):
#     self.epoch +=1
#     self.change = True
#
# def random_invert_img(x):
#   #print(epoch_track.epoch)
#   if epoch_track.epoch >= epochs:
#      return x
#   x_temp = x.numpy()
#   x_temp = x_temp.reshape(x_temp.shape[0], 28,28)
#   x_shifted = []
#   for image in x_temp:
#        x_shifted.append(f.shift_image_np(image))
#   x_shifted = np.array(x_shifted)
#
#   x_result = x_shifted.reshape(x_temp.shape[0],28,28,1)
#
#   return x_result
# def random_invert():
#   return layers.Lambda(lambda x: random_invert_img(x))
#
# random_invert = random_invert()
# class RandomInvert(layers.Layer):
#   def __init__(self, **kwargs):
#     super().__init__(**kwargs)
#
#   def call(self, x):
#     return random_invert_img(x)
#
# epoch_track = Epoch_Tracker()
# tf.random.set_seed(92)
# def model_builder(hp):
#     model = keras.Sequential()
#     hp_units1 = hp.Int('conv1', min_value=40, max_value=140, step=1)
#     hp_units2 = hp.Int('conv2', min_value=40, max_value=100, step=1)
#     hp_units3 = hp.Int('conv3', min_value=32, max_value=80, step=1)
#     hp_drop1 = hp.Int('drop1', min_value=3, max_value=6, step=1)
#     hp_drop2 = hp.Int('drop2', min_value=3, max_value=6, step=1)
#     hp_kernel_size1 = hp.Int('kernel_size1', min_value=3, max_value=7, step=2)
#     hp_kernel_size2 = hp.Int('kernel_size2', min_value=3, max_value=9, step=2)
#     hp_kernel_size3 = hp.Int('kernel_size3', min_value=3, max_value=15, step=2)
#
#
#     # model.add(RandomInvert())
#     # model.add(keras.layers.Conv2D(filters = hp_units1,  kernel_size=(hp_kernel_size1, hp_kernel_size1), padding='same', activation='relu', input_shape=input_shape))
#     # model.add(keras.layers.Conv2D(filters = hp_units2,  kernel_size=(hp_kernel_size2, hp_kernel_size2), padding='same', activation='relu'))
#     # model.add(keras.layers.MaxPool2D())
#     # model.add(keras.layers.Dropout(hp_drop1/10))
#     # model.add(keras.layers.Conv2D(filters = hp_units3,  kernel_size=(hp_kernel_size3, hp_kernel_size3), padding='same', activation='relu'))
#     # model.add(keras.layers.Dropout(hp_drop2/10))
#     # model.add(keras.layers.Flatten())
#     # model.add(keras.layers.Dense(num_classes, activation = 'softmax'))
#     # hp_learning_rate = hp.Choice('learning rate', values=list(range(5,15)))
#     model = tf.keras.models.Sequential([
#         RandomInvert(),
#         tf.keras.layers.Conv2D(hp_units1, (hp_kernel_size1, hp_kernel_size1), padding='same', activation='relu', input_shape=input_shape),
#         tf.keras.layers.Conv2D(hp_units2, (hp_kernel_size1, hp_kernel_size1), padding='same', activation='relu'),
#         tf.keras.layers.MaxPool2D(),
#         tf.keras.layers.Dropout(hp_drop1 / 10),
#         tf.keras.layers.Conv2D(hp_units3, (hp_kernel_size3, hp_kernel_size3), padding='same', activation='relu'),
#         tf.keras.layers.Dropout(hp_drop2 / 10),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(num_classes, activation='softmax')])
#     hp_optimizer = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
#     hp_learning_rate = hp.Choice('learning rate', values=list(range(5, 15)))
#     if hp_optimizer == 'adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate / 10000)
#     elif hp_optimizer == 'nadam':
#         optimizer = tf.keras.optimizers.Nadam(learning_rate=hp_learning_rate / 10000)
#     elif hp_optimizer == 'rmsprop':
#         optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate / 10000)
#
#
#     model.compile(optimizer= optimizer,
#                 loss='categorical_crossentropy',
#                 metrics=['acc'], run_eagerly=True)
#
#     return model
# # @exit_after(28800)
# class MyHyperModel(kt.HyperModel):
#     def search(self, *fit_args, **fit_kwargs):
#         """Performs a search for best hyperparameter configuations.
#         # Arguments:
#             *fit_args: Positional arguments that should be passed to
#               `run_trial`, for example the training and validation data.
#             *fit_kwargs: Keyword arguments that should be passed to
#               `run_trial`, for example the training and validation data.
#         """
#         self.on_search_begin()
#         import time
#         MAX_TIME_ALLOWED = 1000
#         while True:
#             trial_start = time.time()
#             trial = self.oracle.create_trial(self.tuner_id)
#             if trial.status == trial_module.TrialStatus.STOPPED:
#                 # Oracle triggered exit.
#                 tf.get_logger().info('Oracle triggered exit')
#                 break
#             if trial.status == trial_module.TrialStatus.IDLE:
#                 # Oracle is calculating, resend request.
#                 continue
#
#             self.on_trial_begin(trial)
#             self.run_trial(trial, *fit_args, **fit_kwargs)
#             self.on_trial_end(trial)
#             trial.update('elapsed', time.time() - trial_start)
#             if np.sum(trial.metrics.get_history('elapsed')) > MAX_TIME_ALLOWED:
#                 break
#         self.on_search_end()
# def run_search(x_train, y_train, x_val, y_val, project_name, fold_numb):
#     tuner = kt.RandomSearch(model_builder,
#                          objective='val_acc',
#                          max_trials= 5,
#                          directory='random_search',
#                          project_name=project_name)
#
#
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.7)
#     #tensorboard = TensorBoard(log_dir='cross-validation/{}'.format(NAME), update_freq='batch', )
#     tuner.search(x= x_train,y =  y_train, epochs=1, validation_data=(x_val,y_val), callbacks=[early_stop], verbose = 1)
#
#     # Get the optimal hyperparameters
#     print('aaa')
#     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#     # print(best_hps)
#     f.save_evolution_results(number_of_models='', conv1=best_hps.get('conv1'),
#                              conv2=best_hps.get('conv2'), conv3=best_hps.get('conv3'), lr=best_hps.get('learning rate'),
#                              kernel1=best_hps.get('kernel_size1'), kernel2=best_hps.get('kernel_size2'),
#                              kernel3=best_hps.get('kernel_size3'), opt=best_hps.get('optimizer'),
#                              dropout1=best_hps.get('drop1'), dropout2=best_hps.get('drop2'), val_acc='',
#                              number='', fold_numb=fold_numb, time='', file_name='random_results.csv')
#     conv1 = best_hps.get('conv1')
#     print(f'Number of conv blocks: {conv1}')
#     # for hyparam in [f'filters_{i}' for i in range(nblocks)] + [f'pooling_{i}' for i in range(nblocks)] + [
#     #     'Dense units'] + ['learning_rate']:
#     # #     print(f'{hyparam}: {best_hps.get(hyparam)}')
#     # best_model = tuner.get_best_models(1)[0]
#     # best_model.summary()
# # run_search()
# # print(f"""
# # The hyperparameter search is complete. The optimal number of units in the first layer is conv1
# # {best_hps.get('conv1')} and  conv2  {best_hps.get('conv2')} and
# #   conv3  {best_hps.get('conv3')} and  dropu is  {best_hps.get('dense')}'.
# # """)
# # model = tuner.hypermodel.build(best_hps)
# # history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, callbacks = callbacks_list)
# #
# # test_loss, test_acc = model.evaluate(x_test, y_test)
# #
# # history_dict = history.history
# # # Save it under the form of a json file
# # json.dump(history_dict, open('saved_history_cnn_hyperband', 'w'))
#
#
# if __name__ == '__main__':
#     f.save_evolution_results(number_of_models='', conv1='40-140', conv2='40-100', conv3='32-80', lr='5--15',
#                              kernel1='3--7', kernel2='3--9', kernel3='3--15', opt='',
#                              dropout1='3--6', dropout2='3--6', val_acc='', number=0, fold_numb=0, time=0, file_name='random_results.csv')
#     total_time = 5 * 60 * 60  # whole time for running each fold
#     folds_numbers = ['0', '00','1', '2', '3', '4', '5']
#     x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train,
#                                                    x_test, y_test)
#     for fold_numb in folds_numbers:
#         if fold_numb == '1':
#             x_val = x_train[:12000]
#             x_train = x_train[12000:]
#             y_val = y_train[:12000]
#             y_train = y_train[12000:]
#             NAME = "Random_fold1"
#             print(f'\n training for the fold number {fold_numb} \n')
#         elif fold_numb == '2':
#             x_val = x_train[12000:24000]
#             x_train = np.concatenate((x_train[:12000], x_train[24000:]), axis=0)
#             y_val = y_train[12000:24000]
#             y_train = np.concatenate((y_train[:12000], y_train[24000:]), axis=0)
#             NAME = "Random_fold2"
#             print(f'\n training for the fold number {fold_numb} \n')
#         elif fold_numb == '3':
#             x_val = x_train[24000:36000]
#             x_train = np.concatenate((x_train[:24000], x_train[36000:]), axis=0)
#             y_val = y_train[24000:36000]
#             y_train = np.concatenate((y_train[:24000], y_train[36000:]), axis=0)
#             NAME = "Random_fold3"
#             print(f'\n training for the fold number {fold_numb} \n')
#         elif fold_numb == '4':
#             x_val = x_train[36000:48000]
#             x_train = np.concatenate((x_train[:36000], x_train[48000:]), axis=0)
#             y_val = y_train[36000:48000]
#             y_train = np.concatenate((y_train[:36000], y_train[48000:]), axis=0)
#             NAME = "Random_fold4"
#             print(f'\n training for the fold number {fold_numb} \n')
#         elif fold_numb == '5':
#             x_val = x_train[48000:60000]
#             x_train = x_train[:48000]
#             y_val = y_train[48000:60000]
#             y_train = y_train[:48000]
#             NAME = "Random_fold5"
#             print(f'\n training for the fold number {fold_numb} \n')
#         elif fold_numb == '0' or fold_numb == '00':
#             x_val = x_train[:100]
#             x_train = x_train[-100:]
#             y_val = y_train[:100]
#             y_train = y_train[-100:]
#             NAME = "Random_fold0"
#             print(f'\n training for the fold number {fold_numb} \n')
#     # Start foo as a process
#         p = multiprocessing.Process(target=run_search, name="run seacrh", args=(x_train, y_train, x_val, y_val, NAME, fold_numb))
#         p.start()
#         time.sleep(100)
#
#         # Terminate foo
#         p.terminate()
#
#         # Cleanup
#         p.join()
#         print(f'\n finished')
#
import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import numpy as np
from tensorflow.keras import layers

import keras_tuner as kt
from tensorflow import keras
import sys
import threading
import time
from time import sleep

num_classes = 10
input_shape = (28, 28, 1)
tf.random.set_seed(1234)
dataset = 'ORACLE'
x_train_orig, y_train_orig,  x_test_orig, y_test_orig = f.choose_dataset(dataset)
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
size_data = x_train_orig.shape[0]
batch_size = 64
epochs =9
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
tf.random.set_seed(92)
def model_builder(hp):
    model = keras.Sequential()
    hp_units1 = hp.Int('conv1', min_value=40, max_value=140, step=1)
    hp_units2 = hp.Int('conv2', min_value=40, max_value=100, step=1)
    hp_units3 = hp.Int('conv3', min_value=32, max_value=80, step=1)
    hp_drop1 = hp.Int('drop1', min_value=3, max_value=6, step=1)
    hp_drop2 = hp.Int('drop2', min_value=3, max_value=6, step=1)
    hp_kernel_size1 = hp.Int('kernel_size1', min_value=3, max_value=7, step=2)
    hp_kernel_size2 = hp.Int('kernel_size2', min_value=3, max_value=9, step=2)
    hp_kernel_size3 = hp.Int('kernel_size3', min_value=3, max_value=15, step=2)


    model.add(RandomInvert())
    model.add(keras.layers.Conv2D(filters = hp_units1,  kernel_size=(hp_kernel_size1, hp_kernel_size1), padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(filters = hp_units2,  kernel_size=(hp_kernel_size2, hp_kernel_size2), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(hp_drop1/10))
    model.add(keras.layers.Conv2D(filters = hp_units3,  kernel_size=(hp_kernel_size3, hp_kernel_size3), padding='same', activation='relu'))
    model.add(keras.layers.Dropout(hp_drop2/10))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_classes, activation = 'softmax'))
    hp_learning_rate = hp.Choice('learning rate', values=list(range(5,15)))

    hp_optimizer = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
    if hp_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate / 10000)
    elif hp_optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=hp_learning_rate / 10000)
    elif hp_optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate / 10000)


    model.compile(optimizer= optimizer,
                loss='categorical_crossentropy',
                metrics=['acc'], run_eagerly=True)

    return model

def run_search(NAME,x_train, y_train, x_val, y_val, max_trials):
    start_time = time.time()
    tuner = kt.RandomSearch(model_builder,
                         objective='val_acc',
                         max_trials= max_trials,
                         directory='oracle/random_search',
                         project_name=NAME)


    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.7)
    tuner.search(x = x_train, y = y_train, epochs=30, validation_data=(x_val, y_val))

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    # print(best_hps)
    f.save_evolution_results(number_of_models='', conv1=best_hps.get('conv1'),
                             conv2=best_hps.get('conv2'), conv3=best_hps.get('conv3'), lr=best_hps.get('learning rate'),
                             kernel1=best_hps.get('kernel_size1'), kernel2=best_hps.get('kernel_size2'),
                             kernel3=best_hps.get('kernel_size3'), opt=best_hps.get('optimizer'),
                             dropout1=best_hps.get('drop1'), dropout2=best_hps.get('drop2'), val_acc='',
                             number=0, fold_numb=fold_numb, time='', file_name='random_results.csv')
# folds_numbers = ['1']
folds_numbers = ['1', '2', '3', '4', '5']
x_train_orig, y_train_orig, x_test_orig, y_test_orig = f.edit_data(x_train_orig, y_train_orig,
                                                       x_test_orig, y_test_orig)
for fold_numb in folds_numbers:
    if dataset == 'ORACLE':
        folds_train, folds_labels = f.split_dataset(dataset, x_train_orig, y_train_orig)
        for fold_numb in folds_numbers:
            if fold_numb == '1':
                x_train = np.concatenate((folds_train[1], folds_train[2], folds_train[3], folds_train[4]))
                y_train = np.concatenate((folds_labels[1], folds_labels[2], folds_labels[3], folds_labels[4]))
                x_val = folds_train[0]
                y_val = folds_labels[0]
                max_trials = 35
            elif fold_numb == '2':
                x_train = np.concatenate((folds_train[0], folds_train[2], folds_train[3], folds_train[4]))
                y_train = np.concatenate((folds_labels[0], folds_labels[2], folds_labels[3], folds_labels[4]))
                x_val = folds_train[1]
                y_val = folds_labels[1]
                max_trials = 40
            elif fold_numb == '3':
                x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[3], folds_train[4]))
                y_train = np.concatenate((folds_labels[0], folds_labels[1], folds_labels[3], folds_labels[4]))
                x_val = folds_train[2]
                y_val = folds_labels[2]
                max_trials = 45
            elif fold_numb == '4':
                x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[4]))
                y_train = np.concatenate((folds_labels[0], folds_labels[1], folds_labels[2], folds_labels[4]))
                x_val = folds_train[3]
                y_val = folds_labels[3]
                max_trials =50
            elif fold_numb == '5':
                x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[3]))
                y_train = np.concatenate((folds_labels[0], folds_labels[1], folds_labels[2], folds_labels[3]))
                x_val = folds_train[4]
                y_val = folds_labels[4]
                max_trials = 55
            print(f'\n training for the fold number {fold_numb} \n')
            NAME = "Random_fold" + fold_numb
            run_search(NAME, x_train, y_train, x_val, y_val, max_trials)