import tensorflow as tf
from keras.datasets import fashion_mnist

import json
import matplotlib.pyplot as plt
import seaborn as sns
import functions as f
from keras.callbacks import ModelCheckpoint
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
input_shape = (28, 28, 1)

x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
batch_size = 64
num_classes = 10
epochs = 200
tf.random.set_seed(1234)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
log_csv = tf.keras.callbacks.CSVLogger('my_logs.csv', separator = ',', append = False)
filepath = 'saved_model/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [log_csv, early_stop, checkpoint]
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dropout(0.1),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.25,
                    callbacks=callbacks_list,
                    verbose=1
                    )
model.save("fnn_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('saved_history_fnn', 'w'))