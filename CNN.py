import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import json
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from scipy.ndimage.interpolation import shift
import time


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#x_train, y_train = f.shift_x_train(x_train, y_train)
#plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     #plt.xlabel(class_names[train_labels[i][0]])
# plt.show()
input_shape = (28, 28, 1)
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)

size_data = x_train.shape[0]
batch_size = 64
num_classes = 10
num_iter =20000
epochs =int ((batch_size / size_data) * num_iter)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
log_csv = keras.callbacks.CSVLogger('history_cnn_new.csv')
filepath = 'cnn_saved_model/ aug_weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [early_stop, checkpoint,log_csv]

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
#     tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
#     tf.keras.layers.MaxPool2D(strides=(2, 2)),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])
# def random_shift_img(x):
#     dx = random.randint(0, 2)
#     dy = random.randint(0, 2)
#     x = f.shift_image(x, dx, dy)
#     return x
# def random_shift():
#   return layers.Lambda(lambda x: random_shift_img(x))
#
# random_shift = random_shift()
#
# class RandomShift(layers.Layer):
#   def __init__(self, **kwargs):
#     super().__init__(**kwargs)
#
#   def call(self, x):
#     return random_shift_img(x)
def random_invert_img(x):
  x_temp = x.numpy()
  x_temp = x_temp.reshape(x_temp.shape[0], 28,28)
  # x_result = []
  # for i in range(x_temp.shape[0]):
  #     sing_ex = f.shift_image(x_temp[i])
  #     x_result.append(sing_ex)
  # x_result = np.array(x_result)
  x_shifted = f.shift_image_np(x_temp)
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
    #RandomInvert(),
    tf.keras.layers.Conv2D(59, (5, 5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(33, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(4 / 10),
    tf.keras.layers.Conv2D(112, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Dropout(3/ 10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'], run_eagerly=True)
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
model.save("fashion_model_aug.h5")
eval_metrics = model.evaluate(x_test, y_test)
history_dict = history.history

json.dump(history_dict, open('saved_history_aug', 'w'))
