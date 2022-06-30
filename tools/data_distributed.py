import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms


def CIFAR10_distributed_dataloader_generator(batch, num_workers,
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
    test_dataset = datasets.CIFAR10(root='./data',
                                    train=False,
                                    transform=transform_test)

    dataset_size = len(train_dataset)
    valid_size = int(dataset_size * valid_fraction)
    test_size = dataset_size - valid_size

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [test_size, valid_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset,
                                                                    shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset=valid_dataset,
                                                                    shuffle=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True,
                              sampler=valid_sampler)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)

    return train_loader, valid_loader, test_loader
