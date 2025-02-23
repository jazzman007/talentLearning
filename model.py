import torch
import numpy as np
from config import *
import copy

class Model:
    def __init__(self, input_data_size: int, output_data_size: int, dna: dict):
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.dna = dna

        if dna is None:
            n = np.random.randint(1, 10)
            self.dna = {
               # 'num_layers': n,
               # 'layer_sizes': [np.random.randint(1, 10) for i in range(n)]
                'num_layers': 1,
                'layer_sizes': [1],
                'layer_type' : ['Sigmoid']  
            }
        else:
            assert(dna['num_layers'] == len(dna['layer_sizes']))
            self.dna = dna
        layers = []
        layers.append(torch.nn.Linear(self.input_data_size, self.dna['layer_sizes'][0]))
        match self.dna['layer_type'][0]:
            case 'ReLU':
                layers.append(torch.nn.ReLU())
            case 'Sigmoid':
                layers.append(torch.nn.Sigmoid())
            case 'Tanh':
                layers.append(torch.nn.Tanh())
        for i in range(1, self.dna['num_layers']):
            layers.append(torch.nn.Linear(self.dna['layer_sizes'][i-1], self.dna['layer_sizes'][i]))
            match self.dna['layer_type'][i]:
                case 'ReLU': 
                    layers.append(torch.nn.ReLU())
                case 'Sigmoid':
                    layers.append(torch.nn.Sigmoid())
                case 'Tanh':
                    layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.dna['layer_sizes'][-1], self.output_data_size))
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def evaluate(self, x, y):
        return torch.nn.functional.mse_loss(self.forward(x), y)

    def recombine(self, other_model):
        if self.dna['num_layers'] > other_model.dna['num_layers']:
            bigger = self
            smaller = other_model
        else:
            bigger = other_model
            smaller = self
        new_dna = {
            'num_layers': copy.deepcopy(bigger.dna['num_layers']),
            'layer_sizes': copy.deepcopy(bigger.dna['layer_sizes']),
            'layer_type': copy.deepcopy(bigger.dna['layer_type'])
        }
        for i in range(smaller.dna['num_layers']):
            new_dna['layer_sizes'][i] = (bigger.dna['layer_sizes'][i] + smaller.dna['layer_sizes'][i]) // 2
        return Model(self.input_data_size, self.output_data_size, new_dna)

    def train(self, x, y, train_parameters):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_parameters['learning_rate'])
        for epoch in range(train_parameters['num_epochs']):
            optimizer.zero_grad()
            loss = self.evaluate(x, y)
            loss.backward()
            optimizer.step()
            if train_parameters['print_every'] is not None and epoch % train_parameters['print_every'] == 0:
                print(f"Epoch {epoch}, loss {loss.item()}")
    
    def mutate(self):
        which_mutation = np.random.choice(['add_layer', 'remove_layer', 'change_layer_size'], p=[0.4, 0.2, 0.4])
        match which_mutation:
            case 'add_layer':
                return self.add_layer()
            case 'remove_layer':
                return self.remove_layer()
            case 'change_layer_size':
                return self.change_layer_size()

    def add_layer(self):
        # add a layer at random position of size of the left neighboring layer 
        new_dna = {
            'num_layers': self.dna['num_layers'] + 1,
            'layer_sizes': self.dna['layer_sizes'],
            'layer_type': self.dna['layer_type']
        }
        new_layer_position = np.random.randint(self.dna['num_layers'])
        new_dna['layer_sizes'].insert(new_layer_position, self.dna['layer_sizes'][new_layer_position])
        new_dna['layer_type'].insert(new_layer_position, np.random.choice(['ReLU', 'Sigmoid', 'Tanh']))
        return Model(self.input_data_size, self.output_data_size, new_dna)
    
    def remove_layer(self):
        # remove a layer at random position
        if self.dna['num_layers'] == 1:
            return self
        new_dna = {
            'num_layers': self.dna['num_layers'] - 1,
            'layer_sizes': self.dna['layer_sizes'],
            'layer_type': self.dna['layer_type']
        }
        position_to_remove = np.random.randint(new_dna['num_layers'])
        new_dna['layer_sizes'].pop(position_to_remove)
        new_dna['layer_type'].pop(position_to_remove)
        return Model(self.input_data_size, self.output_data_size, new_dna)

    def change_layer_size(self):
        # change size of a random layer by 1
        new_dna = {
            'num_layers': self.dna['num_layers'],
            'layer_sizes': self.dna['layer_sizes'],
            'layer_type': self.dna['layer_type']
        }
        layer_to_mutate = np.random.randint(new_dna['num_layers'])
        if new_dna['layer_sizes'][layer_to_mutate] > 1:
            new_dna['layer_sizes'][layer_to_mutate] += np.random.choice([-1, 1], p=[0.2, 0.8])
        else:
            new_dna['layer_sizes'][layer_to_mutate] += 1

        return Model(self.input_data_size, self.output_data_size, new_dna)

    #when used in print, print the sequntial model configuration
    def __str__(self):
        return str(self.model)
         
        