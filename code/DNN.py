import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt

kernel_size   = 5
stride_length = 1
padding_value = 2
def convLayer(input_channels, output_channels,kernel_size=kernel_size,
        stride_length = stride_length, padding=padding_value, activation=True):
    if activation:
           return nn.Sequential(
               nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size,
                   stride=stride_length, padding=padding_value),
               #nn.BatchNorm2d(output_channels),
               nn.ReLU(),
               nn.MaxPool2d(2)
           )
    else:
           return nn.Sequential(
               nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size,
                   stride=stride_length, padding=padding_value),
               nn.BatchNorm2d(output_channels),
           )

def fcLayer(input_channels, output_channels):
        return nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
        )

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet , self).__init__()
        #Tensor Dimension: (BATCH_SIZE, 1, 28, 28)
        self.conv1 = convLayer(1,16)
        #Tensor Dimension: (BATCH_SIZE, 16, 14, 14)
        self.conv2 = convLayer(16,32) 
        #Tensor Dimension: (BATCH_SIZE, 32,7,  7)
        self.fc1   = fcLayer(32*7*7, 120)
        self.fc2   = fcLayer(120,    84)
        self.out   = nn.Linear(84,     10)
    def forward(self, input_data):
        x_1    = self.conv1(input_data)
        x_2    = self.conv2(x_1)
        x      = x_2.view(x_2.size(0), -1) #flat (batch_size, 32*7*7)
        x      = self.fc1(x)
        x      = self.fc2(x)
        output = self.out(x)
        return output

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):#imagenet数量
        super(AlexNet , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        '''
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=5*5*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        '''
        self.out = nn.Linear(in_features=5*5*256, out_features=num_classes)
    def forward(self, input_data):
        x_1    = self.layer1(input_data)
        x_2    = self.layer2(x_1)
        x_3    = self.layer3(x_2)
        x_4    = self.layer4(x_3)
        x_5    = self.layer5(x_4)
        x      = x_5.view(x_5.size(0), -1) #flat (batch_size, 256*5*5)
        #x      = self.layer6(x)
        #x      = self.layer7(x)
        output = self.out(x)
        return output

class Branch(nn.Module):
    def __init__(self, input_channels, num_classes=10, num_layers=2, AfterConv =
            1024):
        super(Branch, self).__init__()
        self.model = None
        self.out   = None
        convChannels = input_channels
        if num_layers == 1:
            self.model = nn.Sequential(
                convLayer(input_channels = input_channels,
                          output_channels= convChannels, 
                          kernel_size=3),
            )
        elif num_layers == 2:
            convChannels = int(convChannels/2)
            self.model = nn.Sequential(
                convLayer(input_channels = input_channels,
                          output_channels= input_channels, 
                          kernel_size=3),
                convLayer(input_channels = input_channels,
                          output_channels= convChannels, 
                          kernel_size=3),
            )
        self.out = nn.Linear(AfterConv, num_classes)
    def forward(self ,x):
        if self.model is not None:
            x = self.model(x)
        x = x.view(x.size(0), -1) #flat (batch_size, 256*5*5)
        return self.out(x)

import math
def Entropy(x):
    x = F.softmax(x, dim=1)
    T = 0.0
    num = 0
    for eleRow in x:
        num += 1
        for ele in eleRow:
            T += ele * math.log(ele)
    T = T * (-1.0)
    return T/float(num)


class AlexNet_Branchy(nn.Module):
    def __init__(self, num_classes = 10, input_channels=1):#imagenet数量
        super(AlexNet_Branchy, self).__init__()
        '''
            base line: layer1: conv11*11
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


        '''
            branch1:   layer1_1: conv3*3(to simplify: only fcLayer)
                       exit1
        '''
        self.exit1 = Branch(input_channels = 96, num_classes =
                num_classes,num_layers=0,
                AfterConv = 26*26*96)

        '''
            base line: layer2: conv5*5
                       layer3: conv3*3
                       to simplify: only fcLayer
        '''
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
        )

        '''
            branch2:   layer3_1, layer3_2: conv3*3, conv3*3
                       exit2
        '''
        self.exit2 = Branch(input_channels = 384, num_classes = num_classes,
                num_layers = 0, AfterConv= 12*12*384)

        '''
            base line: layer4: conv3*3
                       layer5: conv3*3
        '''
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        '''
            exit3
        '''
        self.out= nn.Sequential(
            nn.Linear(in_features=5*5*256, out_features=1024),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    '''
        useType = 1: testing
        useType = 0: training
    '''
    def forward(self, input_data, useType = 1):
        #training
        if not useType:
            predictions = []
            #base line layer1
            x_1    = self.layer1(input_data)
            #branch 1
            prediction_branch1 = self.exit1(x_1)
    
            #base line layer2,3
            x_2    = self.layer2(x_1)
            x_3    = self.layer3(x_2)
    
            #branch 2
            prediction_branch2 = self.exit2(x_3)

            #base line layer4,5
            x_4    = self.layer4(x_3)
            x_5    = self.layer5(x_4)
    
    
            #base line exit3
            x      = x_5.view(x_5.size(0), -1) #flat (batch_size, 256*5*5)
            exit3  = self.out(x)

            predictions.append(prediction_branch1)
            predictions.append(prediction_branch2)
            predictions.append(exit3)
    
            return predictions
       # testing
        else:
            import math
            T1 = 0.11
            T2 = 0.09
            #base line layer1
            x_1    = self.layer1(input_data)
            #branch 1
            prediction_branch1 = self.exit1(x_1)
            T = Entropy(prediction_branch1)
            exitType = "1"
            if T < T1:
                return exitType, prediction_branch1
    
            #base line layer2,3
            x_2    = self.layer2(x_1)
            x_3    = self.layer3(x_2)
    
            #branch 2
            prediction_branch2 = self.exit2(x_3)
            T = Entropy(prediction_branch2)
            exitType = "2"
            if T < T2:
                return exitType, prediction_branch2

            #base line layer4,5
            x_4    = self.layer4(x_3)
            x_5    = self.layer5(x_4)
    
    
            #base line exit3
            x      = x_5.view(x_5.size(0), -1) #flat (batch_size, 256*5*5)
            exit3  = self.out(x)

            exitType = "3"
            return exitType, exit3

class Squeeze_Layer(nn.Module):
    def __init__(self, input_channels, output_channels, squeeze_num):#imagenet数量
        super(Squeeze_Layer, self).__init__()
        #squeeze 1*1
        self.squeeze_layer = nn.Conv2d(in_channels=int(input_channels),
                out_channels=squeeze_num, kernel_size=1)
        #expand 1*1
        self.expand_layer1 = nn.Conv2d(in_channels=int(squeeze_num),
                out_channels=output_channels//2, kernel_size=1)
        #expand 3*3
        self.expand_layer2 = nn.Conv2d(in_channels=int(squeeze_num),padding=1,
                out_channels=output_channels//2, kernel_size=3)

    def forward(self, input_data):
        x  = self.squeeze_layer(input_data)
        e1 = self.expand_layer1(x)
        e2 = self.expand_layer2(x)
        return torch.cat((e1,e2), 1)

class AlexNet_Squeeze(nn.Module):
    def __init__(self, num_classes = 10):#imagenet数量
        super(AlexNet_Squeeze, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = Squeeze_Layer(96, 256, 32)
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer3 = Squeeze_Layer(256, 384, 48)
        self.layer4 = Squeeze_Layer(384, 384, 48)
        self.layer5 = Squeeze_Layer(384, 256, 64)
        self.pool2  = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=1)
        )
    def forward(self, input_data):
        x    = self.layer1(input_data)
        x    = self.layer2(x)
        x    = self.pool1(x)
        x    = self.layer3(x)
        x    = self.layer4(x)
        x    = self.layer5(x)
        x    = self.pool2(x)
        x    = self.layer6(x)
        x    = nn.functional.avg_pool2d(input=x, kernel_size=5)
        x    = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x
