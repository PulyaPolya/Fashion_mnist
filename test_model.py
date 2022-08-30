import tensorflow as tf
from keras.datasets import fashion_mnist
import functions as f
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#x_train, y_train = f.shift_x_train(x_train, y_train)
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
# x_train = f.opposite(x_train)
# x_test = f.opposite(x_test)
size_data = x_train.shape[0]
batch_size = 64
num_classes = 10
num_iter =20000
epochs =int ((batch_size / size_data) * num_iter)
# plt.figure(figsize=(3, 3))
# plt.imshow(x_train[0], cmap='binary')
# plt.show()

NAME = "aug_np-1-{}".format(int(time.time()))

conv1=59
conv2=33
conv3=112
def random_invert_img(x):
  x_temp = x.numpy()
  x_temp = x_temp.reshape(x_temp.shape[0], 28,28)
  # x_result = []
  # for i in range(x_temp.shape[0]):
  #     sing_ex = f.shift_image(x_temp[i])
  #     x_result.append(sing_ex)
  # x_result = np.array(x_result)
  #x_shifted = f.opposite(x_temp)
  x_shifted = f.shift_image_np(x_temp)
  # plt.figure(figsize=(3, 3))
  # plt.imshow(x_shifted[0], cmap='binary')
  # plt.show()
  x_result = x_shifted.reshape(x_temp.shape[0],28,28,1)
  return x_result
def random_invert():
  return layers.Lambda(lambda x: random_invert_img(x))

random_invert = random_invert()
class RandomInvert(layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, x):
    return random_invert_img(x)



input_shape = (28, 28, 1)
model = tf.keras.models.Sequential([
    RandomInvert(),
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
filepath = 'cnn_saved_model/aug_np_weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]



model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'],  run_eagerly=True)
start = time.time()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=callbacks_list,
                    verbose=1
                    )
end = time.time()
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end-start)))
model.save("aug_np_model.h5")
eval_metrics = model.evaluate(x_test, y_test)