# Tests for the Model class
import torch
from model import Model

def test_init():
    input_data_size = 5
    output_data_size = 6
    model = Model(input_data_size, output_data_size, None)
    assert Model is not None

def test_parameters():
    input_data_size = 5
    output_data_size = 6
    model = Model(input_data_size, output_data_size, None)
    assert model.input_data_size == input_data_size
    assert model.output_data_size == output_data_size

def test_empty_dna():
    input_data_size = 5
    output_data_size = 6
    model = Model(input_data_size, output_data_size, None)
    assert model.dna == {'num_layers': 1, 'layer_sizes': [1]} 

def test_nonempty_dna():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 2, 'layer_sizes': [2, 3]}
    model = Model(input_data_size, output_data_size, dna)
    assert model.dna == dna
    assert input_data_size == model.input_data_size
    assert output_data_size == model.output_data_size

def test_recombine_same_size():
    input_data_size = 5
    output_data_size = 6
    dna1 = {'num_layers': 2, 'layer_sizes': [2, 3]}
    dna2 = {'num_layers': 2, 'layer_sizes': [4, 5]}
    model1 = Model(input_data_size, output_data_size, dna1)
    model2 = Model(input_data_size, output_data_size, dna2)
    new_model = model1.recombine(model2)
    assert new_model.dna['num_layers'] == 2
    assert new_model.dna['layer_sizes'] == [3, 4]

def test_recombine_different_size():
    input_data_size = 5
    output_data_size = 6
    dna1 = {'num_layers': 2, 'layer_sizes': [2, 3]}
    dna2 = {'num_layers': 3, 'layer_sizes': [4, 5, 6]}
    model1 = Model(input_data_size, output_data_size, dna1)
    model2 = Model(input_data_size, output_data_size, dna2)
    new_model = model1.recombine(model2)
    assert new_model.dna['num_layers'] == 3
    assert new_model.dna['layer_sizes'] == [3, 4, 6]