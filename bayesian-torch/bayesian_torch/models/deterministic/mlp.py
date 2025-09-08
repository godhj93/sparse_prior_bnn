import torch

class MLP(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dims, activation=torch.nn.ReLU):
        super(MLP, self).__init__()
        
        self.linear1 = torch.nn.Linear(input_dim, hidden_dims[0])
        self.linear2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = torch.nn.Linear(hidden_dims[1], output_dim)
        self.activation = activation()
        self.bn1 = torch.nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = torch.nn.BatchNorm1d(hidden_dims[1])
    
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x