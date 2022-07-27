import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import json
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
input_shape = (28, 28, 1)
batch_size = 64
num_classes = 10
epochs =100
input_dim = 28
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
model = tf.keras.models.Sequential([

tf.keras.layers.LSTM(128, dropout=0.3, input_shape=(None,input_dim) ,return_sequences=True),
tf.keras.layers.LSTM(64, dropout=0.3, return_sequences=True),
tf.keras.layers.LSTM(64, dropout=0.3),

tf.keras.layers.Dense(num_classes, activation='softmax')

    ])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
log_csv = keras.callbacks.CSVLogger('history_cnn_new.csv')
filepath = 'rnn_saved_model/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [early_stop, checkpoint,log_csv ]
# model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    callbacks=callbacks_list,
                    verbose=1
                    )
model.save("fashion_model_rnn.h5")
test_loss, test_acc = model.evaluate(x_test, y_test)
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('saved_history_rnn', 'w'))