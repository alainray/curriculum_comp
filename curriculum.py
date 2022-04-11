
import torch
from typing import Sized
from torch.utils.data.sampler import Sampler
from typing import List, Sized
from torch.utils.data import DataLoader,IterableDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from collections import defaultdict
import numpy as np
data_length = {'cifar10': 50000, 'cifar100': 50000, 'imagenet': 1024000}
# Learning Strats
# Standard 
# Curriculum
# Anticurriculum
# Banded Curriculum
# Self Paced Learning
# Self Paced Curriculum Learning
# Teacher-Student Curriculum

# Pacing strategies
def log(t,args,**kwargs):
    T = args.iters
    N = data_length[args.train_ds]
    a = args.a
    b = args.b 
    result = N*b + N*(1-b)*(1+0.1*np.log(t/(a*T)+ np.exp(-10)))
    return result if result < N else N

def poly(t,args,p=1,**kwargs):
    T = args.iters
    N = data_length[args.train_ds]
    a = args.a
    b = args.b 
    result = N*b + N*(1-b)/((a*T)**p)*(t**p)
    return result if result < N else N

def linear(t, args):
    return poly(t,args,p=1)
def quad(t, args):
    return poly(t,args,p=2)
def root(t, args):
    return poly(t,args,p=1/2)

def exp(t,args,**kwargs):
    T = args.iters
    N = data_length[args.train_ds]
    a = args.a
    b = args.b 
    result = N*b + N*(1-b)/(np.exp(10)-1)*(np.exp(10*t/(a*T)) - 1)
    return result if result < N else N

def step(t,args,**kwargs):
    T = args.iters
    N = data_length[args.train_ds]
    a = args.a
    b = args.b 
    result = N*b + N*(1-b)*np.floor(t/(a*T))
    return result if result < N else N


pacing = {"log": log, "poly": poly, "exp": exp, "step": step}

def create_schedule(n_steps=10, sched_type="linear", max_iterations=1000, full_iters=None):
    schedule = defaultdict(int)
    if sched_type=="curr":
        if not full_iters:
            full_iters = max_iterations/2
        for i in range(1,n_steps+1):
            schedule[int((max_iterations-full_iters)*i*1.0/float(n_steps))] = i*1.0/float(n_steps)
    elif sched_type == "std":
        schedule[1] = 1.0
    else:
        schedule[1] = 1.0
    return schedule

def curriculize(cl, scores, score_match: List = None, **params):
    r"""
    - cl: Dataset class to instantiate.
    - scores: list of scores. If score_match is not provided, then it is assumed that
    the ith element of dataset matches the difficulty of scores[i].
    - score_match: a List that for a given index i returns the correct index in scores 
    that matches that element.
    - params: params for instantiating the Dataset.
    """
    dataset= cl(**params)
    assert(len(dataset) == len(scores)), "Length of scores do not match length of dataset"
    if score_match is not None:
        dataset.scores = [scores[score_match[i]] for i in range(len(scores))]
    else:
        dataset.scores = scores
    return dataset

def curriculum_loader(dataset, cutoff, args, anticurriculum=False):
    r"""
    - dataset: dataset instance.
    - cutoff: score at which to cut examples.
    - anticurriculum: whether to go from harder to easier examples.
    """
    sampler = CurriculumSampler(dataset, cutoff=cutoff, anticurriculum=anticurriculum)
    return DataLoader(dataset, sampler=sampler, batch_size=args.train_bs)
    
class CurriculumSampler(Sampler):
    data_source: Sized
    r"""
    - data_source
    - cutoff: score at which to cut examples.
    - anticurriculum: whether to go from harder to easier examples.
    """
    def __init__(self, data_source: Sized, cutoff: float, anticurriculum: bool = False) -> None:
        self.data_source = data_source
        self.cutoff = cutoff
        self.anticurriculum = anticurriculum
    
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):
        dataset_size = len(self.data_source.scores)
        #print(f"\nDataset size is: {int(dataset_size*self.cutoff)}")
        indices = np.argsort(self.data_source.scores) # from harder to easier
        if not self.anticurriculum:
            indices = indices[::-1]
        n = int(self.cutoff)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        s = torch.randperm(n, generator=generator)
        yield from indices[s].tolist()
        
    def __len__(self) -> int:
        return int(self.cutoff)


pacing = {"linear": linear, "quad": quad, "root": root, "exp": exp, "step": step, "log": log }

if __name__ == "__main__":
    import numpy as np
    transform = ToTensor()
    ds = CIFAR100(".", transform=transform)
    params = {"root": ".", "transform": transform}
    scores = np.load("c_score/cifar10/scores.npy")
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(seed)
    s = torch.randperm(len(scores), generator=generator)
    print(scores)
    print(scores[s])
    '''
    
    cds = curriculize(CIFAR10,scores,**params)
    print(cds)
    loader_params = {'batch_size': 10}
    dl = curriculum_loader(cds, cutoff=0.9, anticurriculum=False, **loader_params)
    batch = next(iter(dl))

    sched = create_schedule(max_iterations=10000)
    print(sched)'''