'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import sys
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR10

# Utils
import visdom
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from config import *


##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar10_train = CIFAR10('./cifar10', train=True, download=True, transform=train_transform)
cifar10_unlabeled   = CIFAR10('./cifar10', train=True, download=True, transform=test_transform)
cifar10_test  = CIFAR10('./cifar10', train=False, download=True, transform=test_transform)

# train_loader = keep changing
test_loader  = DataLoader(cifar10_test, batch_size=BATCH)


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


##
# Train
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)

        target_loss = criterion(scores, labels)
        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0].detach()
            features[1].detach()
            features[2].detach()
            features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                m_backbone_loss.item(),
                m_module_loss.item(),
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss or Acc',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')

#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            #pred_loss = models['module'](features)
            pred_loss = criterion(scores, labels)
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


##
# Main
if __name__ == '__main__':
    vis = visdom.Visdom(server='http://localhost', port=9000)
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    for trial in range(TRIALS):
        ##
        # Initialize a labeled dataset by randomly sampling K=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]

        train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                                          sampler=SubsetRandomSampler(labeled_set), 
                                          pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):
            ##
            # Model
            resnet18    = resnet.ResNet18().cuda()
            loss_module = lossnet.LossNet().cuda()
            models = {'backbone': resnet18, 'module': loss_module}
            torch.backends.cudnn.benchmark = True

            ##
            # Loss, Criterion and Scheduler Initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            #models = {'backbone': resnet18, 'module': loss_module}
            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}
            #dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

            ##
            # Training and Testing
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
            acc = test(models, dataloaders, mode='test')
            print('Trial {} || Cycle {} || Label set size {}: Test acc {}'.format(trial+1, cycle+1, len(labeled_set), acc))

            ##
            # Update the Labeled Dataset via Loss Prediction-based Uncertainty Measurement
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), 
                                          pin_memory=True)
            uncertainty = get_uncertainty(models, unlabeled_loader) #torch.rand(SUBSET)#
            arg = np.argsort(uncertainty)
            assert len(uncertainty) == 10000
            
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            assert len(labeled_set) % 1000 == 0
            assert len(set(labeled_set)) == len(labeled_set)
            assert len(set(unlabeled_set)) == len(unlabeled_set)
            assert len(labeled_set) + len(unlabeled_set) == 50000
            assert len(set(labeled_set).intersection(set(unlabeled_set))) == 0
            assert len(set(labeled_set).intersection(set(subset))) == 1000

            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
        
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))