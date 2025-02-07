import torch

class Model:
    def __init__(input_data_size: torch.Size, output_data_size: torch.Size, dna: dict):
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.dna = dna

        if dna is None:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_data_size, 1),
                torch.nn.ReLU(),
                torch.nn.Linear(1, output_data_size)
            )
        else:
            self.model = torch.nn.Sequential()
            for i in range(dna['num_layers']):
                torch.nn.Linear(dna['layer_sizes'][i], dna['layer_sizes'][i+1])
                torch.nn.ReLU()
            torch.nn.Linear(dna['layer_sizes'][-1], output_data_size)
    
    def forward(self, x):
        return self.model(x)

    def evaluate(self, x, y):
        return torch.nn.functional.mse_loss(self.forward(x), y)

    def recombine(self, other_model):
        new_num_layers = (len(self.model) + len(other_model.model)) // 2
        new_layer_sizes = [layer.in_features for layer in self.model] + [layer.in_features for layer in other_model.model]
        dna = {
            'num_layers': new_num_layers,
            'layer_sizes': [layer.in_features for layer in self.model]
        }
        return Model(self.input_data_size, self.output_data_size, dna)
                
         
        