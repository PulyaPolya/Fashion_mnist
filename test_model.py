import tensorflow as tf
from keras.datasets import fashion_mnist
import functions as f
from keras.callbacks import ModelCheckpoint, TensorBoard
import time

size_data = 60000
batch_size = 64
num_classes = 10
num_iter =10000
epochs =int ((batch_size / size_data) * num_iter)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train[:size_data], y_train[:size_data], x_test, y_test)
NAME = "60k-{}".format(int(time.time()))

conv1=59
conv2=33
conv3=112

input_shape = (28, 28, 1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(59, (5, 5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(33, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(4 / 10),
    tf.keras.layers.Conv2D(112, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Dropout(3/ 10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')])
tensorboard = TensorBoard(log_dir='experiments/{}'.format(NAME), update_freq='batch',)
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, baseline= 0.5)
filepath = 'cnn_saved_model/60k_weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]



model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
start = time.time()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=callbacks_list,
                    verbose=1
                    )
end = time.time()
print(f'Total elapsed time {end-start}\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end-start)))
model.save("60k_model.h5")
eval_metrics = model.evaluate(x_test, y_test)