import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import json
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from scipy.ndimage.interpolation import shift
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback


def edit_data(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255.0
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    return x_train, y_train, x_test, y_test



def csv_to_json(csvFilePath, jsonFilePath):
    # Open a csv reader called DictReader
    data = {}
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:
            # Assuming a column named 'No' to
            # be the primary key
            key = rows['epoch']
            data[key] = rows

    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
def pandas_heplp():
    dict = {}
    df = pd.read_csv('history.csv')
    num_ep = len(df['epoch'])
    dict["loss"] =[]
    dict["acc"] = []
    dict["val_acc"]=[]
    dict["val_loss"]=[]
    for i in range(num_ep):
         dict["loss"].append(df['loss'][i])
         dict["acc"].append(df['acc'][i])
         dict["val_acc"].append(df['val_acc'][i])
         dict["val_loss"].append(df['val_loss'][i])
    return dict

def unite_dict(dict1, dict2):
    dict3 = {**dict1}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] += dict2[key]
    return (dict3)

def add_zeros(y_train, add_in_begg):
    empt_arr = [0] * 10
    empt_arr = np.array(empt_arr)
    y_train_20 = []
    for i in range(len(y_train)):
        if add_in_begg:
            y_train_20.append(np.concatenate((y_train[i], empt_arr)))
        else:
            y_train_20.append(np.concatenate((empt_arr,y_train[i])))
    y_train_20 = np.array(y_train_20)
    return y_train_20

def get_data_for_d_f(ret = 'all'):
    (x_train_d, y_train_d), (x_test_d, y_test_d) = mnist.load_data()
    (x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()
    x_train_d, y_train_d, x_test_d, y_test_d = edit_data(x_train_d, y_train_d, x_test_d, y_test_d)
    x_train_f, y_train_f, x_test_f, y_test_f = edit_data(x_train_f, y_train_f, x_test_f, y_test_f)

    # y_train_20d = add_zeros(y_train_d, add_in_begg=True)
    # y_train_20f = add_zeros(y_train_f, add_in_begg=False)
    #
    # y_test_20d = add_zeros(y_test_d, add_in_begg=True)
    # y_test_20f = add_zeros(y_test_f, add_in_begg=False)
    y_train_f = np.array(y_train_f)
    y_train_f += 10
    y_test_f = np.array(y_test_f)
    y_test_f += 10
    y_train_20f = tf.one_hot(y_train_f.astype(np.int32), depth=20)
    y_train_20d= tf.one_hot(y_train_d.astype(np.int32), depth=20)
    y_test_20f = tf.one_hot(y_test_f.astype(np.int32), depth=20)
    y_test_20d = tf.one_hot(y_test_d.astype(np.int32), depth=20)
    if ret == 'all':
        y_train = np.concatenate((y_train_20d, y_train_20f))
        y_test = np.concatenate((y_test_20d, y_test_20f))
        x_train = np.concatenate((x_train_d, x_train_f))
        x_test = np.concatenate((x_test_d, x_test_f))
    elif ret == 'd':
        y_train = y_train_20d
        y_test = y_test_20d
        x_train = x_train_d
        x_test = x_test_d
    else:
        y_train = y_train_20f
        y_test = y_test_20f
        x_train = x_train_f
        x_test = x_test_f
    x_train, y_train = shuffle(x_train, y_train)

    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_test, y_test

def transpose(x_train, y_train):
    rotated_images = []
    for image in x_train:
        rotated_image = image.transpose()
        rotated_images.append(rotated_image)
    rotated_images = np.array(rotated_images)
    x_train = np.concatenate((x_train, rotated_images))
    y_train = np.concatenate((y_train, y_train))
    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train

def grayscale_to_rgb(image):
    image = image.reshape((28, 28, 1))
    rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image))
    return rgb

def cnn_best_model(num_classes=10):
    input_shape = (28, 28, 1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def define_cnn_simplified(conv1, conv2,conv3, dropout1, dropout2):
    input_shape = (28, 28, 1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(conv1, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(conv2, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout1 / 10),
        tf.keras.layers.Conv2D(conv3, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Dropout(dropout2 / 10),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')])
    return model


def shift_image(image, dx='NAN', dy='NAN'):
    if dx =='NAN' or dy =='NAN':
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image
def shift_batch_np(x_batch, dx ='nan', dy ='nan'):
    if dx == 'nan'and dy == 'nan':
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1), (0,0)])
    x_shifted = np.pad(x_batch, ((0, 0), (abs(dx), 0), (abs(dy), 0)), mode='constant')
    x_shifted = np.roll(x_shifted, (dx, dy), axis=(1, 2))[:, abs(dx):, abs(dy):]
    return x_shifted

def shift_image_np(x_batch, dx ='nan', dy ='nan'):
    if dx == 'nan'and dy == 'nan':
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1), (0,0)])
    x_shifted = np.pad(x_batch, ((abs(dx), 0), (abs(dy), 0)), mode='constant')
    x_shifted = np.roll(x_shifted, (dx, dy), axis=(0, 1))[abs(dx):, abs(dy):]
    return x_shifted


def shift_batch_not_random(x_batch,epoch_track='NAN', dx = 'NAN', dy = 'NAN', ):
    #array = [(0,0),(1,0), (-1,0), (0,1), (0,-1)]
    array = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]
    if epoch_track != "NAN":
        #dx, dy = array[epoch_track.epoch%5]
        dx, dy = random.choice(array)
        if epoch_track.change:
            print(dx, dy)
            epoch_track.change = False
    else:
        x_batch = np.array(x_batch)
    x_shifted = np.pad(x_batch, ((0, 0), (abs(dx), 0), (abs(dy), 0)), mode='constant')
    x_shifted = np.roll(x_shifted, (dx, dy), axis=(1, 2))[:, abs(dx):, abs(dy):]
    return x_shifted

def shift_x_train(x_train, y_train):
    # x_train_augmented = [image for image in x_train]
    # y_train_augmented = [image for image in y_train]
    x_train_augmented = []
    y_train_augmented= []
    for image, label in zip(x_train, y_train):
            for i in range (5):
                dx, dy = random.choice([(0,0),(1, 0), (-1, 0), (0, 1), (0, -1)])
                x_train_augmented.append(shift_image(image, dx, dy))
                y_train_augmented.append(label)
        # for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
        #      x_train_augmented.append(shift_image(image, dx, dy))
        #      y_train_augmented.append(label)
    x_train_augmented = np.array(x_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    x_train_augmented, y_train_augmented = shuffle(x_train_augmented, y_train_augmented)
    return x_train_augmented, y_train_augmented

def shift_x_train_np_not_random(x_train, y_train):
    # x_train_augmented = [image for image in x_train]
    # y_train_augmented = [image for image in y_train]
    x_train_augmented = []
    y_train_augmented = []
    array = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    i = 0
    # for i in range(5):
    #     dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
    for dx, dy in ((1,0), (-1,0), (0,1), (0,-1), (0,0)):

        shift = list(shift_batch_not_random(x_train,'NAN', dx, dy))
        x_train_augmented+= shift
        y_train_augmented+= list(y_train)
    # for image, label in zip(x_train, y_train):
    #         x_train_augmented.append(shift_image_np(image, dx, dy))
    #         y_train_augmented.append(label)
    x_train_augmented = np.array(x_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    x_train_augmented, y_train_augmented = shuffle(x_train_augmented, y_train_augmented)
    return x_train_augmented, y_train_augmented
def shift_x_train1(x_train, y_train):
        x_train_augmented = [image for image in x_train]
        y_train_augmented = [image for image in y_train]
        # x_train_augmented = []
        # y_train_augmented =[]
        for image, label in zip(x_train, y_train):
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                 for image, label in zip(x_train, y_train):
                         x_train_augmented.append(shift_image(image, dx, dy))
                         y_train_augmented.append(label)
        x_train_augmented = np.array(x_train_augmented)
        y_train_augmented = np.array(y_train_augmented)
        x_train_augmented, y_train_augmented = shuffle(x_train_augmented, y_train_augmented)
        return x_train_augmented, y_train_augmented
def opposite(x):
    return 1-x


def plot(x_train, size = 3, iterable = True):
    if iterable:
        for i in range(size):
            plt.figure(figsize=(3, 3))
            plt.imshow(x_train[i], cmap='binary')
            plt.show()
    else:
        plt.figure(figsize=(3, 3))
        plt.imshow(x_train, cmap='binary')
        plt.show()



def compare_arr(a,b):
  wrong_indexes = []
  s1, s2 = a.shape
  for i in range (s1):
    for j in range (s2):
      if a[i][j] != b[i][j]:
        wrong_indexes.append((i, j))
  return wrong_indexes

def arr_to_csv(arr, i):
    df = pd.DataFrame(arr)
    df.to_csv(f'arr_{i}.csv', mode='a', index=False, header=None)

def save_results(type, iter, time, val_acc, test_acc, number):
    if number == 1:
        header = ['type', '#iter', 'time', 'val_acc', 'test_acc']
    else:
        header = None
    df = pd.DataFrame([[type, iter, time, val_acc, test_acc]],
                     index=[number], columns=['type', '#iter', 'time', 'val_acc', 'test_acc'])
    df.to_csv("results.csv", mode='a', index=True, header=header)


def test_function(epochs, conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer):

    class Epoch_Tracker:
        def __init__(self):
            self.epoch = 0
            self.change = True

        def increase(self):
            self.epoch += 1
            self.change = True

    epoch_track = Epoch_Tracker()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train, y_train, x_test, y_test = edit_data(x_train[:50], y_train[:50], x_test, y_test)

    size_data = x_train.shape[0]
    batch_size = 64
    num_classes = 10
    num_iter = 50000

    def random_invert_img(x):
        # print(epoch_track.epoch)
        if epoch_track.epoch >= epochs:
            return x
        x_temp = x.numpy()
        x_temp = x_temp.reshape(x_temp.shape[0], 28, 28)
        x_shifted = []
        for image in x_temp:
            x_shifted.append(shift_image_np(image))
        x_shifted = np.array(x_shifted)

        x_result = x_shifted.reshape(x_temp.shape[0], 28, 28, 1)
        return x_result

    def random_invert():
        return layers.Lambda(lambda x: random_invert_img(x))

    random_invert = random_invert()

    class RandomInvert(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, x):
            return random_invert_img(x)

    tf.random.set_seed(92)
    input_shape = (28, 28, 1)
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate / 10000)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate / 10000)
    elif optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate / 10000)
    model = tf.keras.models.Sequential([
        RandomInvert(),
        tf.keras.layers.Conv2D(conv1, (kernel1, kernel1), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(conv2, (kernel2, kernel2), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout1 / 10),
        tf.keras.layers.Conv2D(conv3, (kernel3, kernel3), padding='same', activation='relu'),
        tf.keras.layers.Dropout(dropout2 / 10),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')])
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, baseline= 0.5)
    num_ep = LambdaCallback(
        on_epoch_end=lambda epoch, logs: epoch_track.increase())
    callbacks_list = [num_ep]

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], run_eagerly=True)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=callbacks_list,
                        verbose=1
                        )
    val_acc = max(history.history['val_acc'])
    return val_acc


def print_model(conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer):
    print(f'conv1 =  {conv1} '
          f'\n conv2 =  { conv2}'
          f'\n conv3 =  {conv3}'
          f'\n dropout1 =  {dropout1 / 10}'
          f'\n dropout2 =  {dropout2 / 10} '
          f'\n kernel1 =  {kernel1} '
          f'\n kernel2 =  {kernel2} '
          f'\n kernel3 =  {kernel3} '
          f'\n optimizer =  {optimizer} '
          f'\n learning_rate =  {learning_rate} '
    )

def save_evolution_results(number_of_models, conv1, conv2,conv3, lr, kernel1,
                           kernel2, kernel3, opt, dropout1, dropout2, val_acc, number, fold_numb, time, file_name = "results.csv"):
    if number == 0:
        header = ['number_of_models', 'conv1', 'conv2', 'conv3', 'learning_rate', 'kernel1', 'kernel2', 'kernel3', 'opt',
                  'drop1', 'drop2','val_acc', 'fold_numb', 'elapsed_time']
    else:
        header = None
    df = pd.DataFrame([[number_of_models, conv1, conv2,conv3, lr, kernel1, kernel2, kernel3, opt, dropout1, dropout2,val_acc,fold_numb, time ]],
                     index=[number], columns=['number_of_models','conv1', 'conv2', 'conv3', 'learning_rate', 'ker1',
                                              'ker2', 'ker2', 'opt', 'drop1', 'drop2','val_acc','fold_numb', 'elapsed_time' ])
    df.to_csv(file_name, mode='a', index=True, header=header)