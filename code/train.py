from CNN import CNN
from DNN import LeNet
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torch.utils.data as data
from readDATA import readMNIST
import json

LR        = 0.001
EPOCH     = 1
loss_func = nn.CrossEntropyLoss()
'''
@parameter:
    model: choose the model, lenet/alexnet/googlenet ...
    EPOCH: training times
    train_loader: using DataLoader in package torch.utils.data to 
'''
def train(model, train_loader, EPOCH=EPOCH, isBranch = False):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        correctDict = {}
        print("Start Training..s....")
        for epoch in range(EPOCH):
            sumNum      = 0.0
            currentLoss = 0
            correctNum  = 0
            exit_loss = [0.0, 0.0, 0.0]
            wn        = [1.0, 0.7, 0.3]
            for i, (x, y) in enumerate(train_loader):
                sumNum  += len(y)
                input_data = Variable(x)
                label      = Variable(y)
                output = None
                loss = 0
                if not isBranch:
                    output = model(input_data)
                    loss = loss_func(output, label)
                else:
                    output = model(input_data, 0)
                    for it,prediction in enumerate(output):
                        exit_loss[it] = loss_func(prediction, label)
                        loss += exit_loss[it]*wn[it]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if not isBranch:
                    currentLoss += float(loss.data)
                    _,predict=torch.max(output,1)
                    correct=(predict==label).sum()
                    correctNum+= int(correct)
                    print(i,correct,loss.data)
                else:
                    correctDict[i] = []
                    for it,prediction in enumerate(output):
                        _,predict=torch.max(prediction,1)
                        correct=(predict==label).sum()
                        correctDict[i].append(int(correct.item()))
                        print(i," correct: ", correct, " loss: ", exit_loss[it])

            epoch_loss= currentLoss/float(sumNum)
            epoch_correct=correctNum/float(sumNum)
            print("epoch {:d}  epoch loss {:f} epoch_correct {:f}".format(epoch,epoch_loss,epoch_correct))
        print("Finish Training......")
        #fileName = "BranchyNetTrendOnMNIST"
        #BranchyNetTrend = json.dump(correctDict, open(fileName, "w"))

