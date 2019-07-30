from CNN import CNN
from DNN import LeNet, AlexNet, AlexNet_Branchy, AlexNet_Squeeze, AlexNet_Branchy_Squeeze
import torch
from torch import nn
from torch.autograd import Variable
import time
from readDATA import readMNIST
from train import train

'''
@parameter:
    model: choose the trained model
    test_loader: using DataLoader in package torch.utils.data to transform the
                 data
'''
def test(model, test_loader, isBranch):
    print("Start Testing......")
    if isBranch:
        exit1_num     = 0
        exit1_correct = 0
        exit2_num     = 0
        exit2_correct = 0
        exit3_num     = 0
        exit3_correct = 0

    sumNum      = 0.0
    correctNum  = 0
    print(test_loader)
    for i, (x, y) in enumerate(test_loader):
        sumNum  += len(y)
        input_data = Variable(x)
        label      = Variable(y)
        if isBranch:
            exit_type_set, output = model(input_data)
            _,predict=torch.max(output,1)
            if exit_type_set == '1':
                exit1_num += 1
                exit1_correct += (predict==label).sum()
            if exit_type_set == '2':
                exit2_num += 1
                exit2_correct += (predict==label).sum()
            if exit_type_set == '3':
                exit3_num += 1
                exit3_correct += (predict==label).sum()

        else:
            output = model(input_data)
        _,predict=torch.max(output,1)
        correct=(predict==label).sum()
        correctNum+= int(correct)
        if i%100 == 0:
            print(i,correct)
    accuracy =correctNum/float(sumNum)
    print("All Network Accuracy {:f}".format(accuracy))
    if isBranch:
        accuracy = exit1_correct.item()/float(exit1_num)
        print("Exit1 Number, Accuracy :{:d} , {:f}".format(exit1_num, accuracy))
        accuracy = exit2_correct.item()/float(exit2_num)
        print("Exit2 Number, Accuracy :{:d} , {:f}".format(exit2_num, accuracy))
        accuracy = exit3_correct.item()/float(exit3_num)
        print("Exit3 Number, Accuracy :{:d} , {:f}".format(exit3_num, accuracy))
    print("Finish Testing......")

def begin():
    net = AlexNet_Branchy_Squeeze()
    train_loader, test_loader = readMNIST()
    '''
        To change the network,
            change net;
            change parameterFile
            change except: train(isBranch=True/False)
            change test(isBranch=True/False)
    '''
    #Begin Train
    parameterFile = "./BranchySqueezeAlexNet.pkl"
    a = None

    try:
        net.load_state_dict(torch.load(parameterFile))
    except:
        train_start = time.perf_counter()
        train(model=net, train_loader=train_loader, isBranch=True)
        train_finish = time.perf_counter()
        torch.save(net.state_dict(), parameterFile)
        train_elapsed = train_finish - train_start
        print("Train Time:",train_elapsed,"s")

    '''
    dic = net.state_dict()
    for name,params in net.named_parameters():
        print(name)
        print(dic[name].size())
        print(dic[name].numel())
    '''

    #Begin Test
    test_start = time.perf_counter()
    test(net, test_loader, isBranch=True)
    test_finish = time.perf_counter()

    test_elapsed =  test_finish - test_start
    print("Test Time:",test_elapsed,"s")
if __name__ == '__main__':
    begin()
