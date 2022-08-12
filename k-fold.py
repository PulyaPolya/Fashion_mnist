import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
from sklearn.model_selection import KFold
import json
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
num_folds = 3
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
batch_size = 64
epochs =30
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
log_csv = keras.callbacks.CSVLogger('history_cnn_new.csv')
filepath = 'cnn_saved_model/k-fold/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
callbacks_list = [early_stop, checkpoint]
acc_per_fold = []
loss_per_fold = []
input_shape = (28, 28, 1)
for train, test in kfold.split(x, y):
  #model = f.cnn_best_model()
  model =f.cnn_best_model()

  # Compile the model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(x[train], y[train],
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=0.1,
                      callbacks=callbacks_list,
                      verbose=1
                      )

  # Generate generalization metrics
  scores = model.evaluate(x[test], y[test])
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])
  model.save(f'fashion_model_cnn_kfold_{fold_no}.h5')
  history_dict = history.history
  # Save it under the form of a json file
  json.dump(history_dict, open(f'saved_history_cnn_kfold_{fold_no}', 'w'))
  # Increase fold number
  fold_no = fold_no + 1
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')