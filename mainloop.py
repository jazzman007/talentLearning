import torch
import numpy as np
from config import *
from population import *
from model import Model
import math

max_generations = 200
population_size = 50
use_talent = True
talent_drop_percentile = 0.2

breed_parameters = {
    'mutation': {
        'mutation_rate': 0.5
    }
}
select_parameters = {
    'elitism': 0.4,
    'random': 0.
}

train_parameters = {
    'num_epochs': 100,
    'batch_size': 10,
    'print_every': None,
    'learning_rate': 0.01
}

# generate sample data for quadratic regression
x = 6.28*np.random.rand(100, 1)
# make sin(x) as the target function
y = np.sin(x)
# define tensors
x_train = torch.tensor(x[:80], dtype=torch.float32)
y_train = torch.tensor(y[:80], dtype=torch.float32)
x_test = torch.tensor(x[80:], dtype=torch.float32)
y_test = torch.tensor(y[80:], dtype=torch.float32)

input_data_size = x_train.shape[1]
output_data_size = y_train.shape[1]

generation = 1
generation_converged = False
best_result = np.inf

population = generate_population(population_size, input_data_size, output_data_size)

debug_print("Starting training loop")

while (generation <= max_generations) and (not generation_converged):

    print(f"Generation {generation}")
    print("Max layers: ", max([model.dna['num_layers'] for model in population]))
    print("Max layer sizes: ", max([max(model.dna['layer_sizes']) for model in population]))

    if use_talent:
        population = talent_drop(population, talent_drop_percentile, x_train, y_train) 

    train_population(population, x_train, y_train, train_parameters)

    test_results = test_population(population, x_test, y_test)
    best_result = min(test_results)
    best_model = population[test_results.index(best_result)]
    best_model_dna = best_model.dna

    population = select_survivors(population, test_results, select_parameters)

    population = breed_population(population, population_size, breed_parameters)

    generation += 1

    #print("Best model inferences :")
    #for i in range(x_test.shape[0]):
    #    print(f"x: {x_test[i]}, y_m: {best_model.forward(x_test[i])}, y_t: {y_test[i]}")
    print("Best model dna:", best_model_dna)
    print("Best result: ", best_result)