import tensorflow as tf
from keras.datasets import fashion_mnist
import functions as f
from keras.callbacks import ModelCheckpoint, TensorBoard
import time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
NAME = "Evolution-{}".format(int(time.time()))
model = f.define_cnn_simplified(conv1=51, conv2=61, conv3=91, dropout1=4, dropout2=5)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), update_freq='batch',)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, baseline= 0.5)
filepath = 'cnn_saved_model/optimization/ evolution_weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [early_stop, checkpoint, tensorboard]

batch_size = 64
num_classes = 10
epochs =100

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=callbacks_list,
                    verbose=1
                    )

model.save("evolution_model.h5")
eval_metrics = model.evaluate(x_test, y_test)