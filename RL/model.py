import torch as torch 
import torch.nn as nn
import numpy as np
from typing import Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple



class MLP(nn.Module):
    def __init__(self, input_size:int, action_size:int, hidden_size:int=256,non_linear:nn.Module=nn.ReLU):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer

        This is a seperate class because it may be useful for the bonus questions
        """
        super(MLP, self).__init__()
        #====== TODO: ======
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size,action_size)#output layer
        self.non_linear = non_linear()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #====== TODO: ======
        x = x.view(x.size(0), -1)
        x = self.non_linear(self.linear1(x.view(x.size(0), -1)))
        #x = self.output(x)
        return self.output(x)

class Nature_Paper_Conv(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int], action_size: int, **kwargs):
        super(Nature_Paper_Conv, self).__init__()
        #print('input_size:', input_size)  # Example input size: (4, 84, 84)

        self.CNN = nn.Sequential(*[
            nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        ])
        cnn_out_size = self.CNN(torch.zeros(1, *input_size)).shape[1:]
        #self.MLP = MLP(3136,action_size,hidden_size=512)
        self.MLP = MLP(np.prod(cnn_out_size), action_size, hidden_size=512)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.CNN(x)
        # out = out.view(out.size(0), -1)
        # return self.MLP(out)
        return self.MLP(self.CNN(x))



        
    
    


