import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
import json
import functions as f
history_dict = json.load(open('saved_history_fnn', 'r'))
#history_dict = f.pandas_heplp()
filepath = 'cnn_saved_model/ weights-improvement--13-0.20.hdf5'
#filepath = 'cnn_saved_model/transfer/trans_weights-improvement--23-0.21.hdf5'
#filepath = 'fashion_model_cnn.h5'
model = tf.keras.models.load_model(filepath)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train, y_train, x_test, y_test)
#x_train, y_train, x_test, y_test = f.get_data_for_d_f('all')
test_loss, test_acc = model.evaluate(x_test, y_test)

fig, ax = plt.subplots(2,1)
ax[0].plot(history_dict['loss'], color='b', label="Training Loss")
ax[0].plot(history_dict['val_loss'], color='r', label="Validation Loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(history_dict ['acc'], color='b', label="Training Accuracy")
ax[1].plot(history_dict ['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
fig.suptitle('This is a fashion mnist plot', fontsize=16)
plt.show()
Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1)
Y_true = np.argmax(y_test,axis = 1)
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g')
plt.show()