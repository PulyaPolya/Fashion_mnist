from evolution import Evolution
import tensorflow as tf
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import keras
import functions as f
from operator import itemgetter
import json
import timeit
import time
start = time.time()
tf.random.set_seed(1234)
def define_model(conv1, conv2,conv3, dropout1, dropout2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(conv1, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(conv2, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout1 / 10),
        tf.keras.layers.Conv2D(conv3, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Dropout(dropout2 / 10),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')])
    return model

def print_model(conv1, conv2,conv3, dropout1, dropout2):
    print(f'conv1 =  {conv1} '
          f'\n conv2 =  { conv2}'
          f'\n conv3 =  {conv3}'
          f'\n dropout1 =  {dropout1 / 10}'
          f'\n dropout2 =  {dropout2 / 10} '
    )

def train_models(models, numb_iteration, first_run= False, prev_two_val_acc = None):
    val_acc_arr = []
    for hyper_params in models:
        print('-----------------------------------------------')
        print(f'Training for a model number {models.index(hyper_params)+1}, {numb_iteration}th iteration')
        print(f'The parameters are: ')
        print_model(conv1=hyper_params[0], conv2=hyper_params[1], conv3=hyper_params[2],
                             dropout1=hyper_params[3], dropout2=hyper_params[4])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.7)
        model = define_model(conv1=hyper_params[0], conv2=hyper_params[1], conv3=hyper_params[2],
                             dropout1=hyper_params[3], dropout2=hyper_params[4])
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4), loss='categorical_crossentropy', metrics=['acc'])
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            callbacks = [early_stop],
                            verbose=1
                            )
        val_acc = max(history.history['val_acc'])
        val_acc_arr.append(val_acc)
    return val_acc_arr

def get_best_model(val_acc_arr, models):
    index_best_model = val_acc_arr.index(max(val_acc_arr))
    best_model = models[index_best_model].copy()
    return best_model

(x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()
input_shape = (28, 28, 1)
number_of_tr_ex = 10000
x_train, y_train, x_test, y_test = f.edit_data(x_train[:number_of_tr_ex], y_train[:number_of_tr_ex],
                                               x_test, y_test)

batch_size = 64
num_classes = 10
epochs = 10
numb_of_runs = 50
range_dict = {'conv1' : [32, 64], 'conv2' :[32, 64] ,'conv3' : [32, 128],
              'dropout1' : [3, 5], 'dropout2' : [3,5]}
best_acc = -1
evolution = Evolution(range_dict, numb_of_indiv=4)
models = evolution.initialize()
val_acc_arr = train_models(evolution.individuals, 1, first_run= True)
full_models = models[:]
print(val_acc_arr)
for i in range(numb_of_runs):
    if max(val_acc_arr) > best_acc:
        best_acc = max(val_acc_arr)
        best_model = get_best_model(val_acc_arr, full_models)
        best_iteration = i
    print('---------------------------------------------------')
    print(f'Best model so far:')
    print_model(best_model[0], best_model[1], best_model[2], best_model[3], best_model[4])
    print(f'best_val_acc = {round(best_acc, 4)}')
    print(f'best model accomplished on iteration number {best_iteration}')
    print('---------------------------------------------------')
    print(f'training for {i+2}th time')
    models, full_models= evolution.run_evolution(val_acc_arr)
    indexes = evolution.choose_n_val(val_acc_arr)
    two_best_val = list(itemgetter(*indexes)(val_acc_arr))
    val_acc_arr_temp = train_models(models, i+2, first_run= False, prev_two_val_acc=sorted(val_acc_arr[-2:]))
    val_acc_arr = two_best_val + val_acc_arr_temp
    print(val_acc_arr)
    print(full_models)
    print(models)
    end = time.time()
    print(f'Total elapsed time {end-start}\n')
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end-start)))
print(f'Best model so far:')
print_model(best_model[0], best_model[1], best_model[2], best_model[3], best_model[4])
print(f'best_val_acc = {round(best_acc, 4)}')


