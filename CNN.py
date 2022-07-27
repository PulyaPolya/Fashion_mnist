import tensorflow as tf
from keras.datasets import fashion_mnist
import keras
import functions as f
import json
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

batch_size = 64
num_classes = 10
epochs =5
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
log_csv = keras.callbacks.CSVLogger('history_cnn_new.csv')
filepath = 'cnn_saved_model/ weights-improvement--{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose=1,  save_best_only=True, mode='max')
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")
callbacks_list = [early_stop, checkpoint,log_csv ]

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
                    validation_split=0.2,
                    callbacks=callbacks_list,
                    verbose=1
                    )
end = timer()
print(end - start)
model.save("fashion_model_cnn.h5")
test_loss, test_acc = model.evaluate(x_test, y_test)
# Y_pred = model.predict(x_test)
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred,axis = 1)
# # Convert testing observations to one hot vectors
# Y_true = np.argmax(y_test,axis = 1)
# # compute the confusion matrix
# confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)
# #plt.figure(figsize=(10, 8))
# sns.heatmap(confusion_mtx, annot=True, fmt='g')
# plt.show()
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('saved_history_cnn', 'w'))