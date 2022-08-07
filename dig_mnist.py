import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.datasets import  mnist
from keras.datasets import fashion_mnist
import numpy as np
import json
import functions as f
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint

history_dict = json.load(open('saved_history_cnn', 'r'))

x_train, y_train, x_test, y_test = f.get_data_for_d_f('all')

input_shape = (28, 28, 1)
batch_size = 64
num_classes = 20
epochs =300
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
#log_csv = keras.callbacks.CSVLogger('history_cnn_new.csv')
filepath = 'digital_fashion/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')

callbacks_list = [early_stop, checkpoint]

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
start = timer()
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=callbacks_list,
                    verbose=1
                    )
end = timer()
print(end - start)
model.save("fashion_digital_model_cnn.h5")
test_loss, test_acc = model.evaluate(x_test, y_test)

history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('saved_history_cnn', 'w'))