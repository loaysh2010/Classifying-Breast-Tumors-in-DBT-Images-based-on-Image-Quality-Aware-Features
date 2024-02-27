import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
#====================================================#

def load_split_train_test(datadir, batch_size=8):
    train_dir = datadir+'train'
    test_dir  = datadir+'test'

    train_transforms = transforms.Compose([
                                           # transforms.RandomRotation(90, expand=True),
                                           # transforms.RandomRotation(180, expand=True),
                                           # transforms.RandomRotation(270, expand=True),
                                           transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                          ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                         ])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return trainloader, testloader

def load_train_Inbreast(datadir,batch_size=8):

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    return trainloader

def load_train_DDSM(datadir,batch_size=8):

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    all_data = datasets.ImageFolder(datadir, transform=train_transforms)
    val = int(0.3 * len(all_data))
    train = len(all_data)-val
    train_data, test_data = random_split(all_data, [train, val], generator=torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return trainloader,testloader

