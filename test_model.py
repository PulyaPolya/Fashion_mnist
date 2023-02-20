import tensorflow as tf
from keras.datasets import fashion_mnist
import functions as f
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
from tensorflow.keras import layers
from keras.callbacks import LambdaCallback
import numpy as np

class Epoch_Tracker:
  def __init__(self):
     self.epoch = 0
     self.change = True
  def increase(self):
    self.epoch +=1
    self.change = True

epoch_track = Epoch_Tracker()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

start = time.time()
#x_train, y_train= f.shift_x_train_np_not_random(x_train, y_train)
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)

#f.plot(x_train, 6)

size_data = x_train.shape[0]
batch_size = 64
num_classes = 10
num_iter =50000
epochs =int ((batch_size / size_data) * num_iter)
#epochs = 1

NAME = "Bayes weird"

def random_invert_img(x):
  #print(epoch_track.epoch)
  if epoch_track.epoch >= epochs:
      return x
  x_temp = x.numpy()
  x_temp = x_temp.reshape(x_temp.shape[0], 28,28)
  # x_result = []
  # for i in range(x_temp.shape[0]):
  #     sing_ex = f.shift_image(x_temp[i])
  #     x_result.append(sing_ex)
  # x_result = np.array(x_result)
  #x_shifted = f.opposite(x_temp)
  x_shifted = []
  for image in x_temp:
      x_shifted.append(f.shift_image_np(image))
  x_shifted = np.array(x_shifted)
  #x_shifted= f.shift_batch_not_random(x_temp,epoch_track)
  #x_shifted = x_temp[:]
  #f.plot(x_shifted[1], 1, iterable= False)
  #f.plot( x_train_big.reshape(x_train_big.shape[0], 28,28)[0], 0, iterable  = False)

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


tf.random.set_seed(92)
input_shape = (28, 28, 1)
conv1 = 140
conv2 = 100
cov3 = 80
drop1 = 7
drop2 = 3
kernel1 = 4
kernel2=7
kernel3 = 10
l_rate = 1
optimizer =tf.keras.optimizers.Adam(learning_rate=l_rate/10000)
model = tf.keras.models.Sequential([
    RandomInvert(),
    tf.keras.layers.Conv2D(conv1, (kernel1, kernel1), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(conv2, (kernel2, kernel2), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(drop1 / 10),
    tf.keras.layers.Conv2D(cov3, (kernel3, kernel3), padding='same', activation='relu'),
    tf.keras.layers.Dropout(drop2/ 10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')])
tensorboard = TensorBoard(log_dir='testing/{}'.format(NAME), update_freq='batch',)
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, baseline= 0.5)
filepath = 'cnn_saved_model/aug_np_weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
num_ep = LambdaCallback(
    on_epoch_end=lambda epoch,logs: epoch_track.increase())
callbacks_list = [num_ep, checkpoint, tensorboard]



model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'],  run_eagerly=True)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=callbacks_list,
                    verbose=1
                    )
end = time.time()
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end-start)))
eval_metrics = model.evaluate(x_test, y_test)
# f.save_results(type='np.load, layer, shift_np,random, not vect, for each dx dy ', iter = 20000,
# #                time = time.strftime("%H:%M:%S", time.gmtime(end-start)), val_acc =history.history['val_acc'][-1],
# #                test_acc = eval_metrics[1], number=7)