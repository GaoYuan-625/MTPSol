import torch
import torch.nn as nn

from torch.nn import functional as F
class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='gelu'):  
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), self.activation, nn.Dropout(self.dropout)]

        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]

        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        y = self.layers(x)
        return y

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class QWenMLP(nn.Module):
    def __init__(self, f_in, f_out, hidden_dim=256, dropout=0.1,):
        super().__init__()
        self.w1 = nn.Linear(f_in, hidden_dim // 2, bias=False)
        self.w2 = nn.Linear(f_in, hidden_dim // 2, bias=False)
        ff_dim_in = hidden_dim // 2
        self.c_proj = nn.Linear(ff_dim_in, f_out, bias=False)

        self._initialize_weights()

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output
    
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
