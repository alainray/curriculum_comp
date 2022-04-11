import comet_ml
import torch
from models import get_model
from datasets import get_train_data, get_test_loader, preproc
from utils import AverageMeter, checkpoint, get_args
from opt import get_opt, get_criterion
from train import train, test
from utils import setup_comet, set_random_state
from curriculum import curriculum_loader, create_schedule, pacing
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
# Handling parameters to experiments
args = get_args()
# Set randomness
set_random_state(args)
# Create datasets and dataloaders
train_data = get_train_data(args)
schedule = create_schedule(max_iterations=args.iters,sched_type=args.method)
trainfull_dl = DataLoader(train_data, batch_size=args.test_bs, shuffle=False) 
if args.method in ['curr','anti']:
    train_dl = curriculum_loader(train_data, args.b, args, args.method!="curr")
else:
    train_dl = DataLoader(train_data, batch_size=args.train_bs, shuffle=True)
test_dl = get_test_loader(args)
# Define model parameters
model = get_model(args)
# Define optimizer
opt = get_opt(args, model)
# Define criterion
criterion = get_criterion(args)
# Training setup
model.to(args.device)
# Comet.ml logging
exp = setup_comet(args)
exp.log_parameters({k:w for k,w in vars(args).items() if "comet" not in k})
model.comet_experiment_key = exp.get_key() # To retrieve existing experiment

# Training
print(f"Training Schedule is: {schedule}")

iteration = 1
loss_meter = AverageMeter()
acc_meter = AverageMeter()
acc5_meter = AverageMeter()

while iteration <= args.iters:
    # check if change in schedule
    if args.method!="std":
        # Redetermine difficulty (SPL, SPCL)
        # recalculate data loader
        data_amount = int(pacing[args.pacing](iteration,args))
        print(f"\nNew Schedule starts with {data_amount} data samples.")
        train_dl = curriculum_loader(train_data, data_amount, args)    
    acc_meter.reset()
    loss_meter.reset()

    for x, label in train_dl:
        bs = x.shape[0]
        opt.zero_grad()
        x = x.to(args.device)
        label = label.to(args.device)
        logits = model(x)
        loss = criterion(logits, label)
        loss.backward()
        loss_meter.update(loss.detach().cpu(),bs)
        opt.step()
        preds = logits.argmax(dim=1)
        correct = (preds == label).cpu().sum()
        acc_meter.update(correct/float(bs),bs)
        print(f"\r[TRAIN] Iter: {iteration:04d}-Loss: {loss_meter.avg:.3f} Acc: {100*acc_meter.avg:.2f}%", end="")
        metrics = dict()
        metrics['loss'] = loss_meter.avg
        metrics['acc'] = 100*acc_meter.avg
        exp.log_metrics(metrics, prefix="train", step=iteration, epoch=iteration)

        if iteration % args.test_every == 0:
            print("")
            test(exp, args, model, test_dl, criterion, iteration, prefix="test")
            print("")
            test(exp, args, model, trainfull_dl, criterion, iteration, prefix="full_train")
            #checkpoint(args, model, stats, iteration, split="train")'''
            print("")
        iteration+=1
        if iteration > args.iters: break
