import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
import json
import functions as f
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint

#history_dict = json.load(open('saved_history_cnn', 'r'))

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)

input_shape = (28, 28, 1)
batch_size = 64
num_classes = 20
epochs =300
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
filepath = 'digital_fashion/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')

callbacks_list = [early_stop, checkpoint]
# model = f.cnn_best_model()
# model.compile(optimizer=tf.keras.optimizers.Adam(
#     learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     validation_split=0.1,
#                     callbacks=callbacks_list,
#                     verbose=1
#                     )
filepath = 'digital_fashion/ weights-improvement--32-0.02.hdf5'
model = tf.keras.models.load_model(filepath)
model.save("digital_model_cnn.h5")
test_loss, test_acc = model.evaluate(x_test, y_test)

#history_dict = history.history
# Save it under the form of a json file
#json.dump(history_dict, open('saved_history_fashion_mnist', 'w'))