
import logging
logging.basicConfig(level=logging.WARNING)

import argparse
from keras.datasets import fashion_mnist
import hpbandster.core.nameserver as hpns
import hpbandster.visualization as hpvis
import functions as f
import numpy as np
from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker
from Worker import MyWorker
import hpbandster.core.result as hpres
import time

# dataset = 'ORACLE'
dataset = 'FASHION'
x_train_orig, y_train_orig,  x_test_orig, y_test_orig = f.choose_dataset(dataset)

f.save_evolution_results(number_of_models = '' ,conv1='40-140', conv2='40-100', conv3='32-80', lr='5--15',
                         kernel1='3--7', kernel2='3--9', kernel3='3--15', opt='',
                         dropout1='3--6',dropout2='3--6', val_acc='', number=0,fold_numb=0, time = 0, file_name = 'bohb_fashion_results.csv')
def run(max,x_train,y_train, x_val, y_val, fold_numb ):
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=2)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=max)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=1)
    parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='bohb')
    args=parser.parse_args()


    # Step 1: Start a nameserver
    # Every run needs a nameserver. It could be a 'static' server with a
    # permanent address, but here it will be started for the local machine with the default port.
    # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
    # Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.

    w = MyWorker(x_train, y_train,x_val, y_val,nameserver='127.0.0.1',run_id='example1')
    w.run(background=True)

    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run BOHB, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.
    result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)

    bohb = BOHB( configspace = w.get_configspace(),
                  run_id = 'example1', nameserver='127.0.0.1',
                  result_logger=result_logger,
                  min_budget=args.min_budget, max_budget=args.max_budget
               )
    start = time.time()
    res = bohb.run(n_iterations=args.n_iterations)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    end = time.time()
    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'
          ''%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))

    result = hpres.logged_results_to_HBS_result('bohb/')

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    # inc_test_loss = inc_run.info['test accuracy']

    print('Best found configuration:')
    print(inc_config)
    print(inc_config['dropout_rate1'])
    print('It achieved accuracies of %f (validation) '%(1-inc_loss))
    elapsed_time = end-start

    f.save_evolution_results(number_of_models=max, conv1=inc_config['num_filters_1'],
                                     conv2=inc_config['num_filters_2'], conv3=inc_config['num_filters_3'], lr=inc_config['lr'],
                                     kernel1=inc_config['num_kernel_1'], kernel2=inc_config['num_kernel_2'],
                                     kernel3=inc_config['num_kernel_3'], opt=inc_config['optimizer'],
                                     dropout1=inc_config['dropout_rate1'], dropout2=inc_config['dropout_rate2'], val_acc=1-inc_loss,
                                     number=92, fold_numb=fold_numb, time= elapsed_time/3600, file_name='bohb_fashion_results.csv')

#
# x_train_orig, y_train_orig, x_test_orig, y_test_orig = f.edit_data(x_train_orig, y_train_orig,
#                                                        x_test_orig, y_test_orig)
folds_train, folds_labels = f.split_dataset(dataset, x_train_orig, y_train_orig)
folds_numbers = ['1', '2', '3', '4', '5']
# folds_numbers = ['1']
for fold_numb in folds_numbers:
    if fold_numb == '1':
        x_train = np.concatenate((folds_train[1], folds_train[2], folds_train[3], folds_train[4]))
        y_train = np.concatenate((folds_labels[1],folds_labels[2], folds_labels[3], folds_labels[4]))
        x_val = folds_train[0]
        y_val = folds_labels[0]
        # x_val = x_train_orig[-70:]
        # y_val = y_train_orig[-70:]
        # x_train = x_train_orig[:70]
        # y_train = y_train_orig[:70]
        max_epochs = 40
    elif fold_numb == '2':
        x_train = np.concatenate((folds_train[0], folds_train[2], folds_train[3], folds_train[4]))
        y_train = np.concatenate((folds_labels[0], folds_labels[2], folds_labels[3], folds_labels[4]))
        x_val = folds_train[1]
        y_val = folds_labels[1]
        max_epochs = 50
    elif fold_numb == '3':
        x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[3], folds_train[4]))
        y_train = np.concatenate((folds_labels[0], folds_labels[1], folds_labels[3], folds_labels[4]))
        x_val = folds_train[2]
        y_val = folds_labels[2]
        max_epochs = 60
    elif fold_numb == '4':
        x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[4]))
        y_train = np.concatenate((folds_labels[0], folds_labels[1], folds_labels[2], folds_labels[4]))
        x_val = folds_train[3]
        y_val = folds_labels[3]
        max_epochs = 70
    elif fold_numb == '5':
        x_train = np.concatenate((folds_train[0], folds_train[1], folds_train[2], folds_train[3]))
        y_train = np.concatenate((folds_labels[0], folds_labels[1], folds_labels[2], folds_labels[3]))
        x_val = folds_train[4]
        y_val = folds_labels[4]
        max_epochs = 80
    print(f'\n training for the fold number {fold_numb} \n')
    NAME = "SH_fold" + fold_numb
    run(max_epochs,x_train, y_train,x_val, y_val,fold_numb )