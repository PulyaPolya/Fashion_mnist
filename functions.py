import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import json
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from scipy.ndimage.interpolation import shift
import random
import matplotlib.pyplot as plt


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
    for i in range(5):
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
    #for dx, dy in ((1,0), (-1,0), (0,1), (0,-1), (0,0)):

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