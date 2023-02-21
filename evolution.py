import random
from operator import itemgetter
class Evolution:
    def __init__(self, numb_of_indiv):
        self.ranges_dict = {'conv1' : [40, 140], 'conv2' :[40, 100] ,'conv3' : [32, 80], 'kernel1' : [3,5,7],
                            'kernel2': [3,5,7, 9],'kernel3' : [3,5,7,9,11, 13,15],
              'dropout1' : [3,6], 'dropout2' : [3,6], 'learning_rate': [5, 15], 'optimizer' : ['nadam','rmsprop', 'adam'],}          #provides a range for each parameter
        self.params = []
        self.numb_of_indiv = numb_of_indiv      # max number of models
        self.individuals = []                   # array containing all models
        self.numb_of_trained_models = 0

    def initialize(self):
        for i in range(self.numb_of_indiv):
            individual = []
            for elem in self.ranges_dict:
                if elem != 'optimizer' and elem != 'kernel1' and elem != 'kernel2' and elem != 'kernel3':
                    param = random.randint(self.ranges_dict[elem][0], self.ranges_dict[elem][1])
                else:
                    param = random.choice(self.ranges_dict[elem])
                individual.append(param)
            self.individuals.append(individual)
        return self.individuals

    def choose_parents(self, val_acc_arr):
        parents = random.choices(self.individuals, weights = val_acc_arr, k =2)
        return parents


    def mutation(self, individual, numb_of_mut=2):
        for i in range(numb_of_mut):
            param_name = random.choices(list(self.ranges_dict.keys()))[0]  # choosing key (parameter name) from dictionary
            index = list(self.ranges_dict.keys()).index(param_name)  # randomly changing one value
            if param_name != 'optimizer' and param_name!= 'kernel1' and param_name != 'kernel2' and param_name != 'kernel3':
                individual[index] = random.randint(self.ranges_dict[param_name][0], self.ranges_dict[param_name][1])
            else:
                individual[index] = random.choice(self.ranges_dict[param_name])
        return individual


    def crossover_func(self, parents):
        index = random.randrange(1,len(parents[0]))        # randomly slicing two parents and swapping second parts
        offspring1 = parents[0][0:index] + parents[1][index:]
        offspring2 = parents[1][0:index] + parents[0][index:]
        return offspring1, offspring2

    def choose_n_val(self,val_acc_arr, n = 2):
        indexes = sorted(range(len(val_acc_arr)), key=lambda sub: val_acc_arr[sub])[-n:]
        return indexes
    def choose_n_best(self, val_acc_arr, n = 2):
        indexes = self.choose_n_val(val_acc_arr, n)
        best = itemgetter(*indexes)(self.individuals)
        return best

    def run_evolution(self, val_acc_arr):
        next_generation= list(self.choose_n_best(val_acc_arr))
        for i in range(int(len(self.individuals) / 2)-1):
            parents = self.choose_parents(val_acc_arr)
            offspring1, offspring2 = self.crossover_func(parents)
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            next_generation += [offspring1, offspring2]
        self.individuals = next_generation.copy()
        return (next_generation[2:], next_generation)




