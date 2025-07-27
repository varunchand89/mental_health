import torch
import torch.nn as nn
import torch.nn.functional as F


class DepressionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(DepressionClassifier, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        # Final output layer: 1 neuron for binary classification
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1 for binary
        
        self.model = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.model(x)