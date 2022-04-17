from typing import Any, Tuple
import torch.utils.data as nn
from os.path import join
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from curriculum import curriculize


def make_data_params(dataset, split):
    if dataset == "imagenet":
        root = "/workspace1/araymond/ILSVRC2012/train/"
    else:
        root = "."
    
    return {'transform': preproc(dataset,split),
            'root': root,
            'train': split == "train",
            'download': True
            }

def get_test_loader(args):
    data = get_datasets(args)
    test_params = make_data_params(args.train_ds, "test")
    test_data = data[args.train_ds](**test_params)
    return DataLoader(test_data, batch_size=args.test_bs)

def get_train_data(args):
    train_data = None
    data = get_datasets(args)
    score = np.load(f"c_score/{args.train_ds}/scores.npy")
    train_params = make_data_params(args.train_ds, "train")
    if args.method in ['curr', 'anti']:  
        train_data = curriculize(data[args.train_ds],score,**train_params)
    else:
        train_data = data[args.train_ds](**train_params)
    return train_data

def make_path(path):
    folder = path.split("_")[0]
    return join(folder, path)

# CIFAR10

'''
Transforms and Data Parameters
'''
def get_datasets(args):

    data = {"imagenet": ImageNet,
            "cifar10": CIFAR10,
            "cifar100": CIFAR100}
    return data


def preproc(dataset, split):
    t = []
    if dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    if dataset == "imagenet":
        t.append(transforms.Resize(256))
        if split == "train":
            t.extend([transforms.RandomResizedCrop(224),
                      transforms.RandomHorizontalFlip()])
        else:
            t.extend([transforms.CenterCrop(224)])
    else:
        if split == "train":
            t.extend([transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(15)])
    t.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(t)



if __name__ == "__main__":
    dataset = "cifar100"
    split = "train" 
    indices = np.load(f"c_score/{dataset}/indices_{split}.npy")
    print(indices)