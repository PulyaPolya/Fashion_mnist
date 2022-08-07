import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import json
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.datasets import fashion_mnist

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