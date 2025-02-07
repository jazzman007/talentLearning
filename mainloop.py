import numpy as np

max_generations = 100
population_size = 100

generation = 1
generation_converged = False
best_result = np.inf

population = generate_population(population_size)

while (generation <= max_generations) and (not generation_converged):
    talents = evaluate_talent(population)
    population = select_survivors(population, talents, test_results)
    population = breed_population(population, population_size)

    train_population(population)
    test_results = test_population(population)
    
    generation += 1