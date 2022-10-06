import tensorflow as tf
from keras.datasets import fashion_mnist
import functions as f
import math
import random
from operator import itemgetter


num_classes = 10
input_shape = (28, 28, 1)
tf.random.set_seed(1234)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train, x_test, y_test = f.edit_data(x_train[:10], y_train[:10], x_test, y_test)
size_data = x_train.shape[0]
batch_size = 64
num_iter =20000
epochs =int ((batch_size / size_data) * num_iter)
class S:
    def __init__(self):
        self.ranges_dict = {'conv1': [32, 64], 'conv2': [32, 64], 'conv3': [32, 128],
                            'dropout1': [3, 5], 'dropout2': [3, 5], 'kernel1':[1,4], 'kernel2':[1,7], 'kernel3':[2,10], 'learning_rate': [5, 15],
                            'optimizer': ['sgd', 'rmsprop', 'adam'], }
        self.best_model = {'val_acc': 0, 'conv1': 1, 'conv2': 1, 'conv3': 1,
                            'dropout1': 1, 'dropout2': 1, 'kernel1':1, 'kernel2':1, 'kernel3':1, 'learning_rate': 1,
                            'optimizer': 'a', }

    def draw_hyper_params(self, numb_conf):
        arr_configs = []
        for i in range (numb_conf):
            config = []
            for elem in self.ranges_dict:
                if elem != 'optimizer':
                    param = random.randint(self.ranges_dict[elem][0], self.ranges_dict[elem][1])
                else:
                    param = random.choice(self.ranges_dict[elem])
                config.append(param)
            arr_configs.append(config)
        return arr_configs

    def select_n_best(self, n, val_acc):
        import pandas as pd
        lst = pd.Series(val_acc)
        i = lst.nlargest(n)
        result = i.index.values.tolist()
        return result

def run_models(arr_configs,epochs):

    val_acc_arr = []
    try:
        len(arr_configs[0])
        for hyper_params in arr_configs:
            conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer = hyper_params
            print('-----------------------------------------------')
            print(f'Training for a model number {arr_configs.index(hyper_params)}')
            print(f'The parameters are: ')
            f.print_model(conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer)
            val_acc = f.test_function(epochs,conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer)
            val_acc_arr.append(val_acc)
    except Exception:
        conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer = arr_configs
        print('-----------------------------------------------')
        print(f'Training for a model number {1}')
        print(f'The parameters are: ')
        f.print_model(conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer)
        val_acc = f.test_function(epochs, conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3,
                                  learning_rate, optimizer)
        val_acc_arr.append(val_acc)

    return val_acc_arr
def rewrite_best_model(inst_s, new_best, new_val):
    conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer = new_best
    inst_s.best_model = {'val_acc': new_val, 'conv1': conv1, 'conv2': conv2, 'conv3': conv3,
                            'dropout1': dropout1, 'dropout2': dropout2, 'kernel1':kernel1, 'kernel2':kernel2,
                         'kernel3':kernel3, 'learning_rate': learning_rate,
                            'optimizer': optimizer, }


def print_best_model(inst_s, arr_configs,val_acc_arr):
    max_val_acc = max(val_acc_arr)
    if inst_s.best_model['val_acc'] < max_val_acc:
        index = val_acc_arr.index(max_val_acc)
        rewrite_best_model(inst_s, arr_configs[index], max_val_acc)
    print(f'Best model so far:')
    _, conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer = \
        itemgetter('val_acc','conv1','conv2', 'conv3','dropout1', 'dropout2', 'kernel1', 'kernel2',
                         'kernel3', 'learning_rate','optimizer' )(inst_s.best_model)
    f.print_model(conv1, conv2, conv3, dropout1, dropout2, kernel1, kernel2, kernel3, learning_rate, optimizer)
    print(f'best_val_acc = {round(max_val_acc, 4)}')

def hypeband(mu):
    inst_s = S()
    s = 4
    R = mu**s
    B = (s+1)*R
    n = math.ceil(B*(mu**s)/(R*(s+1)))
    r = R/ (mu**s)
    arr_configs = inst_s.draw_hyper_params(n)
    for i in range(s+1):
        ni = math.floor(n/(mu**i))
        ri = int(r*(mu**i))
        val_acc_arr=   run_models(arr_configs,ri)
        print_best_model(inst_s, arr_configs, val_acc_arr)
        indexes_of_best = inst_s.select_n_best(math.floor(ni/mu),val_acc_arr)
        if indexes_of_best:
            arr_configs = itemgetter(*indexes_of_best)(arr_configs)
        else:
            return inst_s.best_model



hypeband(3)