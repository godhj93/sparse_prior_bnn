import torch
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.models.deterministic.mlp import MLP

class MLP_uni(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dims, prior_type, activation=torch.nn.ReLU):
        super(MLP_uni, self).__init__()
        
        const_bnn_prior_parameters = {
            'prior_mu': 0.0,
            'prior_sigma': 1.0,
            'prior_type': prior_type,
            'posterior_mu_init': 0.0,
            'posterior_rho_init': -3.0,
            'type': 'Reparameterization',
            'moped_enable': False,
            'moped_delta': 0.5,
            
        }
        
        self.base_model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, activation=activation)
        dnn_to_bnn(self.base_model, const_bnn_prior_parameters)
        
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        out = self.base_model(x)
        kl = get_kl_loss(self.base_model)
        return out, kl