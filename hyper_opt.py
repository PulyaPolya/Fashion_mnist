import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import json
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
import keras_tuner as kt
from tensorflow import keras

num_classes = 10
input_shape = (28, 28, 1)
number_of_tr_ex = 10000
#(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train[:number_of_tr_ex], y_train[:number_of_tr_ex], x_test, y_test)
# img_train = x_train.astype('float32') / 255.0
# img_test = img_test.astype('float32') / 255.0
def model_builder(hp):
  model = keras.Sequential()


  hp_units1 = hp.Int('conv1', min_value=16, max_value=64, step=5)
  hp_units2 = hp.Int('conv2', min_value=32, max_value=128, step=10)
  hp_units3 = hp.Int('conv3', min_value=32, max_value=128, step=10)
  hp_drop1 = hp.Int('drop1', min_value=3, max_value=6, step=1)
  hp_drop2 = hp.Int('drop1', min_value=3, max_value=6, step=1)
  model.add(keras.layers.Conv2D(filters = hp_units1,  kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
  model.add(keras.layers.Conv2D(filters = hp_units2,  kernel_size=(5, 5), padding='same', activation='relu'))
  model.add(keras.layers.MaxPool2D())
  model.add(keras.layers.Dropout(hp_drop1/10))
  model.add(keras.layers.Conv2D(filters = hp_units3,  kernel_size=(5, 5), padding='same', activation='relu'))
  model.add(keras.layers.Dropout(hp_drop2/10))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

  #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
                loss='categorical_crossentropy',
                metrics=['acc'])

  return model
tuner = kt.Hyperband(model_builder,
                     objective='val_acc',
                     max_epochs=10, # number of searches
                     factor=3,
                     directory='my_dir_new',
                     project_name='intro_to_kt3')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
log_csv = keras.callbacks.CSVLogger('history_cnn_new.csv')
filepath = 'cnn_saved_model/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [early_stop, checkpoint]
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is conv1  {best_hps.get('conv1')} and  conv2  {best_hps.get('conv2')} and  conv3  {best_hps.get('conv3')} and  dense is  {best_hps.get('dense')} the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, callbacks = callbacks_list)

test_loss, test_acc = model.evaluate(x_test, y_test)

history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('saved_history_cnn_hyperband', 'w'))

