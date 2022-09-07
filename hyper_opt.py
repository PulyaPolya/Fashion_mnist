import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import numpy as np
import json
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
import keras_tuner as kt
from tensorflow import keras

num_classes = 10
input_shape = (28, 28, 1)
tf.random.set_seed(1234)
x_train = np.load('x_train_shifted.npy.')
y_train = np.load('y_train_shifted.npy.')
x_test = np.load('x_test.npy')
y_test = np.load("y_test.npy")
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
size_data = x_train.shape[0]
batch_size = 64
num_iter =20000
epochs =int ((batch_size / size_data) * num_iter)
#(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()



def model_builder(hp):
  model = keras.Sequential()
  hp_units1 = hp.Int('conv1', min_value=32, max_value=128, step=1)
  hp_units2 = hp.Int('conv2', min_value=32, max_value=128, step=1)
  hp_units3 = hp.Int('conv3', min_value=32, max_value=128, step=1)
  hp_drop1 = hp.Int('drop1', min_value=3, max_value=5, step=1)
  hp_drop2 = hp.Int('drop2', min_value=3, max_value=5, step=1)
  hp_kernel_size1 = hp.Int('kernel_size1', min_value=1, max_value=10, step = 1)
  hp_kernel_size2 = hp.Int('kernel_size2', min_value=1, max_value=10, step=1)
  hp_kernel_size3 = hp.Int('kernel_size3', min_value=1, max_value=10, step=1)

  model.add(keras.layers.Conv2D(filters = hp_units1,  kernel_size=(hp_kernel_size1, hp_kernel_size1), padding='same', activation='relu', input_shape=input_shape))
  model.add(keras.layers.Conv2D(filters = hp_units2,  kernel_size=(hp_kernel_size2, hp_kernel_size2), padding='same', activation='relu'))
  model.add(keras.layers.MaxPool2D())
  model.add(keras.layers.Dropout(hp_drop1/10))
  model.add(keras.layers.Conv2D(filters = hp_units3,  kernel_size=(hp_kernel_size3, hp_kernel_size3), padding='same', activation='relu'))
  model.add(keras.layers.Dropout(hp_drop2/10))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['acc'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_acc',
                     max_epochs= 21, #max epochs in the end
                     directory='hyper_search_whole_dataset',
                     project_name='hyper_whole_dataset')


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.7)
tuner.search(x_train, y_train, epochs=epochs, validation_split=0.1, callbacks=[early_stop])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first layer is conv1
# {best_hps.get('conv1')} and  conv2  {best_hps.get('conv2')} and
#   conv3  {best_hps.get('conv3')} and  dropu is  {best_hps.get('dense')}'.
# """)
# model = tuner.hypermodel.build(best_hps)
# history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, callbacks = callbacks_list)
#
# test_loss, test_acc = model.evaluate(x_test, y_test)
#
# history_dict = history.history
# # Save it under the form of a json file
# json.dump(history_dict, open('saved_history_cnn_hyperband', 'w'))

