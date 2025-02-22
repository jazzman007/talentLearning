import torch
import numpy as np
from config import *
from population import *
from model import Model

max_generations = 100
population_size = 40
use_talent = False
talent_drop = 0.1

breed_parameters = {
    'mutation': {
        'mutation_rate': 0.2
    }
}
select_parameters = {
    'elitism': 0.2,
    'random': 0.1
}

train_parameters = {
    'num_epochs': 20,
    'batch_size': 10,
    'loss_function': torch.nn.functional.mse_loss
}

# generate sample data for quadratic regression
x = np.random.rand(100, 1)
y = x * x 

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

    if use_talent:
        talent_drop(population, talent_drop, x, y) 

    train_population(population, x_train, y_train, train_parameters)

    test_results = test_population(population, x_test, y_test)
    best_result = min(test_results)
    best_model = population[test_results.index(best_result)]
    best_model_dna = best_model.dna

    population = select_survivors(population, test_results, select_parameters)

    population = breed_population(population, population_size, breed_parameters)

    generation += 1

    print("Best model inferences :")
    for i in range(x_test.shape[0]):
        print(f"x: {x_test[i]}, y_m: {best_model.forward(x_test[i])}, y_t: {y_test[i]}")
    print("Best model dna:", best_model_dna)
    print("Best result: ", best_result)