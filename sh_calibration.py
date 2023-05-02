import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import numpy as np
from tensorflow.keras import layers
import time
import keras_tuner as kt
from tensorflow import keras
import oracle as oracle

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
#@exit_after(30)

def run_search(NAME,x_train, y_train, x_val, y_val, num_of_runs, max_epochs):
    for i in range(num_of_runs):
        start = time.time()
        tuner = kt.Hyperband(model_builder,
                             objective='val_acc',
                             max_epochs=  max_epochs,
                             hyperband_iterations=1,
                             directory='oracle/SH',
                             project_name=NAME)




        tuner.search(x = x_train, y = y_train, epochs=10, validation_data=(x_val, y_val))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        end = time.time()
        elapsed_time = end - start
        # print(best_hps)
        f.save_evolution_results(number_of_models=max_epochs, conv1=best_hps.get('conv1'),
                                 conv2=best_hps.get('conv2'), conv3=best_hps.get('conv3'), lr=best_hps.get('learning rate'),
                                 kernel1=best_hps.get('kernel_size1'), kernel2=best_hps.get('kernel_size2'),
                                 kernel3=best_hps.get('kernel_size3'), opt=best_hps.get('optimizer'),
                                 dropout1=best_hps.get('drop1'), dropout2=best_hps.get('drop2'), val_acc='',
                                 number=92, fold_numb=fold_numb, time= elapsed_time/3600, file_name='sh_oracle_results.csv')

    # Get the optimal hyperparameters
folds_numbers = ['1', '2', '3', '4', '5']
# folds_numbers = ['1']
f.save_evolution_results(number_of_models = '' ,conv1='40-140', conv2='40-100', conv3='32-80', lr='5--15',
                         kernel1='3--7', kernel2='3--9', kernel3='3--15', opt='',
                         dropout1='3--6',dropout2='3--6', val_acc='', number=0,fold_numb=0, time = 0, file_name = 'sh_oracle_results.csv')
x_train_orig, y_train_orig, x_test_orig, y_test_orig = f.edit_data(x_train_orig, y_train_orig,
                                                       x_test_orig, y_test_orig)
if dataset == 'FASHION':
    for fold_numb in folds_numbers:
        if fold_numb == '1':
            # x_val = x_train_orig[:12000]
            # x_train = x_train_orig[12000:]
            # y_val = y_train_orig[:12000]
            # y_train = y_train_orig[12000:]
            x_val = x_train_orig[-100:]
            y_val = y_train_orig[-100:]
            x_train = x_train_orig[:100]
            y_train = y_train_orig[:100]

        elif fold_numb == '2':
            x_val = x_train_orig[12000:24000]
            x_train = np.concatenate((x_train_orig[:12000],x_train_orig[24000:] ), axis = 0)
            y_val = y_train_orig[12000:24000]
            y_train = np.concatenate((y_train_orig[:12000],y_train_orig[24000:]), axis=0)
        elif fold_numb == '3':
            x_val = x_train_orig[24000:36000]
            x_train = np.concatenate((x_train_orig[:24000], x_train_orig[36000:]), axis=0)
            y_val = y_train_orig[24000:36000]
            y_train = np.concatenate((y_train_orig[:24000], y_train_orig[36000:]), axis=0)
        elif fold_numb == '4':
            x_val = x_train_orig[36000:48000]
            x_train = np.concatenate((x_train_orig[:36000], x_train_orig[48000:]), axis=0)
            y_val = y_train_orig[36000:48000]
            y_train = np.concatenate((y_train_orig[:36000], y_train_orig[48000:]), axis=0)
        elif fold_numb == '5':
            x_val = x_train_orig[48000:60000]
            x_train = x_train_orig[:48000]
            y_val = y_train_orig[48000:60000]
            y_train = y_train_orig[:48000]
        max_epochs = 103
        print(f'\n training for the fold number {fold_numb} \n')
        NAME = "SH_fold" + fold_numb
        run_search(NAME, x_train, y_train, x_val, y_val, 1, max_epochs)

elif dataset == 'ORACLE':
    folds_train, folds_labels = f.split_dataset(dataset, x_train_orig, y_train_orig)
    for fold_numb in folds_numbers:
        if fold_numb == '1':
            x_train = np.concatenate((folds_train[1], folds_train[2], folds_train[3], folds_train[4]))
            y_train = np.concatenate((folds_labels[1],folds_labels[2], folds_labels[3], folds_labels[4]))
            x_val = folds_train[0]
            y_val = folds_labels[0]
            # x_val = x_train_orig[-100:]
            # y_val = y_train_orig[-100:]
            # x_train = x_train_orig[:100]
            # y_train = y_train_orig[:100]
            max_epochs = 180
        elif fold_numb == '2':
            x_train = np.concatenate((folds_train[0], folds_train[2], folds_train[3], folds_train[4]))
            y_train = np.concatenate((folds_labels[0],folds_labels[2], folds_labels[3], folds_labels[4]))
            x_val = folds_train[1]
            y_val = folds_labels[1]
            max_epochs = 190
        elif fold_numb == '3':
            x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[3], folds_train[4]))
            y_train = np.concatenate((folds_labels[0],folds_labels[1], folds_labels[3], folds_labels[4]))
            x_val = folds_train[2]
            y_val = folds_labels[2]
            max_epochs = 200
        elif fold_numb == '4':
            x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[4]))
            y_train = np.concatenate((folds_labels[0],folds_labels[1], folds_labels[2], folds_labels[4]))
            x_val = folds_train[3]
            y_val = folds_labels[3]
            max_epochs = 210
        elif fold_numb == '5':
            x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[3]))
            y_train = np.concatenate((folds_labels[0],folds_labels[1], folds_labels[2], folds_labels[3]))
            x_val = folds_train[4]
            y_val = folds_labels[4]
            max_epochs =220
        print(f'\n training for the fold number {fold_numb} \n')
        NAME = "SH_fold" + fold_numb
        run_search(NAME, x_train, y_train, x_val, y_val, 1, max_epochs)