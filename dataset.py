import os
import torch
from torchvision import datasets, transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

svhn_mean= (0.4376821, 0.4437697, 0.47280442) 
svhn_std= (0.19803012, 0.20101562, 0.19703614) 

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


class Dataset():
    def __init__(self, path:str, dataset:str, train:bool):
        

        if train:
            if dataset!='imagenet':
                transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),  
                                transforms.ToTensor(),])
            else:
                transform = transforms.Compose([
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),  
                                transforms.ToTensor(),])
                            # transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
        else:    
            transform = transforms.Compose([     
                            transforms.ToTensor(),])
                            # transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
        assert dataset is not None
        if dataset=='CIFAR10':
            dataset = datasets.CIFAR10(root=path, train=train, transform=transform, download=True)
        elif dataset=='CIFAR100':
            dataset = datasets.CIFAR100(root=path, train=train, transform=transform, download=True)
        elif dataset=='SVHN':
            split = 'train' if train else 'test'
            dataset = datasets.SVHN(root=path, split=split, transform=transform, download=True)
        
        elif dataset == 'imagenet':
            if train:
                dataset = datasets.ImageFolder(
                    '../data/imagenet/train/',
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]))
            else:
                dataset = datasets.ImageFolder(
                    '../data/imagenet/val/',
                    transforms.Compose([
                        transforms.ToTensor(),
                    ]))
        elif dataset == 'tiny':
            if train:
                dataset = datasets.ImageFolder(
                    '../data/tiny/train/',
                    transforms.Compose([
                        transforms.RandomCrop(64, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]))
            else:
                dataset = datasets.ImageFolder(
                    '../data/tiny/test/',
                    transforms.Compose([
                        transforms.ToTensor(),
                    ]))
        
        self.dataset = dataset


    def get_dataloader(self, batch_size=32, shuffle=False):

        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    