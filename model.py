import torch
import numpy as np

class Model:
    def __init__(self, input_data_size: int, output_data_size: int, dna: dict):
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.dna = dna

        if dna is None:
            self.dna = {
                'num_layers': 1,
                'layer_sizes': [1]
            }
        else:
            self.dna = dna
        layers = []
        layers.append(torch.nn.Linear(self.input_data_size, self.dna['layer_sizes'][0]))
        layers.append(torch.nn.ReLU())
        for i in range(1, self.dna['num_layers']):
            layers.append(torch.nn.Linear(self.dna['layer_sizes'][i-1], self.dna['layer_sizes'][i]))
            layers.append(torch.nn.ReLU())
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
            'num_layers': bigger.dna['num_layers'],
            'layer_sizes': bigger.dna['layer_sizes']
        }
        for i in range(smaller.dna['num_layers']):
            new_dna['layer_sizes'][i] = (bigger.dna['layer_sizes'][i] + smaller.dna['layer_sizes'][i]) // 2
        return Model(self.input_data_size, self.output_data_size, new_dna)
    def train(self, x, y, train_parameters):
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(train_parameters['num_epochs']):
            optimizer.zero_grad()
            loss = self.evaluate(x, y)
            loss.backward()
            optimizer.step()
    
    def mutate(self):
        new_dna = {
            'num_layers': self.dna['num_layers'],
            'layer_sizes': self.dna['layer_sizes']
        }
        # change size of a random layer by 1
        layer_to_mutate = np.random.randint(new_dna['num_layers'])
        if new_dna['layer_sizes'][layer_to_mutate] > 1:
            new_dna['layer_sizes'][layer_to_mutate] += np.random.choice([-1, 1, 1])
        else:
            new_dna['layer_sizes'][layer_to_mutate] += 1

        return Model(self.input_data_size, self.output_data_size, new_dna)
                
         
        