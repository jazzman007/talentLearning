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
    assert model.dna == {'num_layers': 1, 'layer_sizes': [1], 'layer_type': ['Sigmoid']} 

def test_nonempty_dna():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    model = Model(input_data_size, output_data_size, dna)
    assert model.dna == dna
    assert input_data_size == model.input_data_size
    assert output_data_size == model.output_data_size

def test_nonempty_dna_layers():
    input_data_size = 1
    output_data_size = 1
    dna = {'num_layers': 1, 'layer_sizes': [2], 'layer_type': ['ReLU']}
    model = Model(input_data_size, output_data_size, dna)
    assert len(model.model) == 3

def test_recombine_same_size():
    input_data_size = 5
    output_data_size = 6
    dna1 = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    dna2 = {'num_layers': 2, 'layer_sizes': [4, 5], 'layer_type': ['ReLU', 'Sigmoid']}
    model1 = Model(input_data_size, output_data_size, dna1)
    model2 = Model(input_data_size, output_data_size, dna2)
    new_model = model1.recombine(model2)
    assert new_model.dna['num_layers'] == 2
    assert new_model.dna['layer_sizes'] == [3, 4]

def test_recombine_different_size():
    input_data_size = 5
    output_data_size = 6
    dna1 = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    dna2 = {'num_layers': 3, 'layer_sizes': [4, 5, 6], 'layer_type': ['ReLU', 'Sigmoid', 'Tanh']}
    model1 = Model(input_data_size, output_data_size, dna1)
    model2 = Model(input_data_size, output_data_size, dna2)
    new_model = model1.recombine(model2)
    assert new_model.dna['num_layers'] == 3
    assert new_model.dna['layer_sizes'] == [3, 4, 6]

def test_add_layer():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    model = Model(input_data_size, output_data_size, dna)
    new_model = model.add_layer()
    assert new_model.dna['num_layers'] == 3
    assert (new_model.dna['layer_sizes'] == [2, 2, 3] or new_model.dna['layer_sizes'] == [2, 3, 3])

def test_add_layer_self():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    model = Model(input_data_size, output_data_size, dna)
    model = model.add_layer()
    assert model.dna['num_layers'] == 3
    assert (model.dna['layer_sizes'] == [2, 2, 3] or model.dna['layer_sizes'] == [2, 3, 3])

def test_add_layer_single():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 1, 'layer_sizes': [3], 'layer_type': ['ReLU']}
    model = Model(input_data_size, output_data_size, dna)
    new_model = model.add_layer()
    assert new_model.dna['num_layers'] == 2
    assert new_model.dna['layer_sizes'] == [3, 3]

def test_remove_layer():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    model = Model(input_data_size, output_data_size, dna)
    new_model = model.remove_layer()
    assert new_model.dna['num_layers'] == 1
    assert (new_model.dna['layer_sizes'] == [2] or new_model.dna['layer_sizes'] == [3])
    assert (new_model.dna['layer_type'] == ['ReLU'] or new_model.dna['layer_type'] == ['Sigmoid'])

def test_remove_layer_empty():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 1, 'layer_sizes': [4], 'layer_type': ['ReLU']}
    model = Model(input_data_size, output_data_size, dna)
    new_model = model.remove_layer()
    assert new_model.dna['num_layers'] == 1
    assert new_model.dna['layer_sizes'] == [4]
    assert new_model.dna['layer_type'] == ['ReLU']

def test_change_layer_size():
    input_data_size = 5
    output_data_size = 6
    dna = {'num_layers': 2, 'layer_sizes': [2, 3], 'layer_type': ['ReLU', 'Sigmoid']}
    model = Model(input_data_size, output_data_size, dna)
    new_model = model.change_layer_size()
    assert new_model.dna['num_layers'] == 2
    assert (new_model.dna['layer_sizes'] == [2, 4] or \
            new_model.dna['layer_sizes'] == [2, 2] or \
            new_model.dna['layer_sizes'] == [1, 3] or \
            new_model.dna['layer_sizes'] == [3, 3])
    assert new_model.dna['layer_type'] == ['ReLU', 'Sigmoid']