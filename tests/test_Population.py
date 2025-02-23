# test for the population module
import pytest
import numpy as np
from population import *
from model import Model
from config import *

def test_generate_population():
    population_size = 10
    input_data_size = 5
    output_data_size = 6
    population = generate_population(population_size, input_data_size, output_data_size)
    assert len(population) == population_size
    for model in population:
        assert model.input_data_size == input_data_size
        assert model.output_data_size == output_data_size
        print(model.dna)
        assert model.dna is not None

def test_select_survivors_elitism():
    input_data_size = 5
    output_data_size = 6
    population_size = 10
    population = [Model(input_data_size, output_data_size, dna = {'num_layers' : 1, 'layer_sizes' : [i], 'layer_type' : ['ReLU']}) for i in range(population_size)]
    test_results = np.random.rand(population_size)
    select_parameters = {
        'elitism': 0.3,
        'random': 0.
    }
    correct_survivor_indices = np.argsort(test_results)[:int(select_parameters['elitism'] * population_size)]
    survivors = select_survivors(population, test_results, select_parameters)
    assert len(survivors) == int(select_parameters['elitism'] * population_size)
    for i in range(1, len(survivors)):
        assert survivors[i].dna['layer_sizes'][0] == correct_survivor_indices[i]
    
def test_population():
    assert True