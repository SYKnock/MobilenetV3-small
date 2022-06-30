import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np


def CIFAR10_dataloader_generator(batch, num_workers=1,
                                 valid_fraction=0.2, 
                                 transform_train=None,
                                 transform_test=None):
    if transform_train is None:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))])
        
    if transform_test is None:
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     transform=transform_train,
                                     download=True)
    valid_dataset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     transform=transform_test)
    test_dataset = datasets.CIFAR10(root='./data',
                                    train=False,
                                    transform=transform_test)
    
    dataset_size = len(train_dataset)
    idx_list = list(range(dataset_size))
    np.random.shuffle(idx_list)

    split = int(valid_fraction * dataset_size)

    train_idx = idx_list[split:]
    valid_idx = idx_list[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch,
                              num_workers=num_workers,
                              drop_last=True,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch,
                              num_workers=num_workers,
                              drop_last=True,
                              sampler=valid_sampler)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader
