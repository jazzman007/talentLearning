from model import Model
from config import *
import numpy as np

def generate_population(population_size, input_data_size, output_data_size):
    population = []
    for i in range(population_size):
        population.append(Model(input_data_size, output_data_size, None))
    return population

def evaluate_talent(population, x_train, y_train):
    talents = []
    for model in population:
        talents.append(model.loss(x_train, y_train))
    return talents

def select_survivors(population, test_results, select_parameters):
    debug_print("Selecting survivors")
    # Sort the population by test results
    population = [x for _, x in sorted(zip(test_results, population), key=lambda pair: pair[0])]
    # Select the best models
    num_elites = int(select_parameters['elitism'] * len(population))
    num_random = int(select_parameters['random'] * len(population))
    survivors = population[:num_elites]
    survivors += population[-num_random:]
    debug_print(f"Selected {len(survivors)} survivors")
    return survivors

def breed_population(population, population_size, breed_parameters):
    # breed models
    new_population = []
    # apply recombination
    if len(population) < population_size:
        for i in range(population_size - len(population)):
            parent1 = population[np.random.randint(len(population))]
            parent2 = population[np.random.randint(len(population))]
            new_population.append(parent1.recombine(parent2))
            debug_print(f"Recombined {parent1.dna} and {parent2.dna}")

    # apply mutation
    for i in range(len(new_population)):
        if np.random.rand() < breed_parameters['mutation']['mutation_rate']:
            new_population[i] = new_population[i].mutate()
            debug_print(f"Mutated {new_population[i].dna}")
    
    return population + new_population

def train_population(population, x_train, y_train, train_parameters):
    for model in population:
        model.train(x_train, y_train, train_parameters)
    return population

def test_population(population, x_test, y_test):
    test_results = []
    for model in population:
        test_results.append(model.evaluate(x_test, y_test))
    return test_results