# encoding: utf-8
BATCH_SIZE = 50
def readMNIST():
    import torchvision
    from torchvision import transforms
    import torch.utils.data as data
    train_data = torchvision.datasets.MNIST(
        root='./MNIST/',
        train=True, #training set
        #transform=torchvision.transforms.ToTensor(),#converts a PIL.Image or numpy.ndarray
        transform = torchvision.transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
        download=True
    )
    test_data  = torchvision.datasets.MNIST(
        root='./MNIST/',
        train=False, #training set
        #transform=torchvision.transforms.ToTensor(),#converts a PIL.Image or numpy.ndarray
        transform = torchvision.transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
        download=True
    )
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True)
    test_loader  = data.DataLoader(dataset=test_data, batch_size=1,shuffle=True)
    return train_loader, test_loader
