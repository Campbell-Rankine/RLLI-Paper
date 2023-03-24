import torch as T
import torch.nn as nn

class Learnable_Weighting(nn.Module):
    def __init__(self, in_dim, activation):
        super(Learnable_Weighting, self).__init__()
        assert(callable(activation))

        self.in_d = in_dim
        self.activation = activation

        self.l1 = nn.Linear(self.in_d, self.in_d, bias=False)

    def forward(self, x):
        return self.activation(self.l1(x))
    
    def _unsupervised_loss(self, x):
        """
        Loss function for weighted methods
        """
        raise NotImplementedError