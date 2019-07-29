import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,28,28)
            nn.Conv2d(in_channels=1, #input height 
                      out_channels=16, #n_filter
                     kernel_size=5, #filter size
                     stride=1, #filter step
                     padding=2
                     ), #output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), #output shape (32,7,7)
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))
        self.out = nn.Linear(32*7*7,10)
        
    def forward(self, input_data):
        x_1 = self.conv1(input_data)
        x_2 = self.conv2(x_1)
        x = x_2.view(x_2.size(0), -1) #flat (batch_size, 32*7*7)
        output = self.out(x)
        return output
