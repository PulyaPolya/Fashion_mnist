import random
from operator import itemgetter
class Evolution:
    def __init__(self, ranges_dict, numb_of_indiv):
        self.ranges_dict =ranges_dict           #provides a range for each parameter
        self.params = []
        self.numb_of_indiv = numb_of_indiv      # max number of models
        self.individuals = []                   # array containing all models

    def initialize(self):
        for i in range(self.numb_of_indiv):
            individual = []
            for elem in self.ranges_dict:
                param = random.randint(self.ranges_dict[elem][0], self.ranges_dict[elem][1])
                individual.append(param)
            self.individuals.append(individual)
        return self.individuals

    def choose_parents(self, val_acc_arr):
        parents = random.choices(self.individuals, weights = val_acc_arr, k =2)
        return parents


    def mutation(self, individual):
        param_name = random.choices(list(self.ranges_dict.keys()))[0]  # choosing key (parameter name) from dictionary
        index =list(self.ranges_dict.keys()).index(param_name)        # randomly changing one value
        individual[index] =  random.randint(self.ranges_dict[param_name][0], self.ranges_dict[param_name][1])
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
        return next_generation[2:]



# evol = Evolution({'a':[0,10],'b': [20,30], 'c':[40,50]}, 6)
# evol.initialize()
# print(evol.individuals)
# evol.run_evolution(val_acc_arr=[0.6, 0.4, 1, 0.5, 0.3, 0.2])
# print(evol.individuals)
# evol.run_evolution(val_acc_arr=[0.5, 0.2, 0.1, 0.35, 0.9, 0.7])
# print(evol.individuals)


