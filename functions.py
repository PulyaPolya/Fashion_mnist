import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import json
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

