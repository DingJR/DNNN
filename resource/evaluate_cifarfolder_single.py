import argparse
import torch
from tqdm import tqdm

import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import os,shutil



def test(model, test_dataset, test_loader, num_devices):
    fn0, label0 = test_dataset.imgs[0]
    print(fn0)
    print(label0)
    fn1, label1 = test_dataset.imgs[1]
    print(fn1)
    print(label1)
    model.eval()
    model_losses = [0]*(num_devices + 1)
    num_correct = [0]*(num_devices + 1)
    print('hello,world')
    for k, (data, target) in enumerate(tqdm(test_loader)):
        print(k)
        fn, label = test_dataset.imgs[k]
        print(fn)
        print(label)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        predictions = model(data)
        #print('target is..', target)
        #print('predictions is...', predictions)
        for i, prediction in enumerate(predictions):
            loss = F.cross_entropy(prediction, target, size_average=False).item()
            pred = prediction.data.max(1, keepdim=True)[1]
            correct = (pred.view(-1) == target.view(-1)).long().sum().item()
            num_correct[i] += correct
            model_losses[i] += loss
            if ((i == 6)and(correct == 0)):
               shutil.copy(fn,'./cannot/')
    N = len(test_loader.dataset)
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss / N)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['dev-{}: {:.4f}%'.format(i, 100. * (correct / N))
                        for i, correct in enumerate(num_correct[:-1])])
    print('Test  Loss:: {}, cloud-{:.4f}'.format(loss_str, model_losses[-1] / N))
    print('Test  Acc.:: {}, cloud-{:.4f}'.format(acc_str, 100. * (num_correct[-1] / N)))

    return model_losses, num_correct

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', default='folder', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/cifar10_folder.pth',
                        help='output directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data

    # 类别名称
    class_names = train_dataset.classes
    print('class_names:{}'.format(class_names))

    # 类别名称
    class_names = test_dataset.classes
    print('class_names:{}'.format(class_names))

    x, _ = train_loader.__iter__().next()
    num_devices = x.shape[1]
    in_channels = x.shape[2]
    model = torch.load(args.model_path)
    test(model, test_dataset, test_loader, num_devices)
