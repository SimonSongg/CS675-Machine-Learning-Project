# -*- coding: utf-8 -*-

from __future__ import print_function, division

import copy
import time
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import model_attention_lastest
import test_CSV
import data_age_latest

parser = argparse.ArgumentParser(description = "Alzheimer's Disease Classification: Train")

parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory for storing data_set')
parser.add_argument('--model', type=str, default='pretrained', help='model used for training (pretrained or age or attention')
parser.add_argument('--model_path', type=str, default='./models/current.pth', help='Path for storing model')
parser.add_argument('--batch_size',  type=int,   default=16,    help='Batch size, number of speakers per batch')
parser.add_argument('--weighted_loss',  type=bool,  default=False,  help='Whether to use weighted cross entropy loss')
parser.add_argument('--lr', type=float, default=0.001,  help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,  help='Momentum in SGD')
parser.add_argument('--epoches',type=int,   default=50,    help='Maximum number of epochs')
args = parser.parse_args()


warnings.filterwarnings("ignore")
writer = SummaryWriter()
plt.ion()  # interactive mode


path = args.dataset_dir
bs = args.batch_size
lr = args.lr
momentum = args.momentum
epoches = args.epoches
model_path = args.model_path


# Preparing dataset
if args.model == 'pretrained':
    train_data = test_CSV.DealDataset(path, 'train')
    val_data = test_CSV.DealDataset(path, 'val')
else:
    train_data = data_age_latest.DealDataset(path, 'train')
    val_data = data_age_latest.DealDataset(path, 'val')


train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=bs)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': train_data.__len__(), 'val': val_data.__len__()}
print(dataset_sizes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall = 0.0
    best_acc = 0.0
    # Start training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print('now' + phase)
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                print('now' + phase)
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_confusion = np.zeros([2, 2])
            if args.model == 'pretrained':
                for inputs, labels in dataloaders[phase]:
                    # Send data to GPU (if available)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, age)

                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics recording
                    running_loss += loss.item()
                    y_hat = preds.cpu().numpy()
                    y = labels.cpu().numpy()
                    running_confusion += confusion_matrix(y_hat, y, labels=[1, 0])
            else:
            # Iterate over data.
                for inputs, age, labels in dataloaders[phase]:
                    # Send data to GPU (if available)
                    inputs = inputs.to(device)
                    age = age.to(device)
                    labels = labels.to(device)
                    age = torch.tensor(age, dtype=torch.float32)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, age)

                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics recording
                    running_loss += loss.item()
                    y_hat = preds.cpu().numpy()
                    y = labels.cpu().numpy()
                    running_confusion += confusion_matrix(y_hat, y, labels=[1, 0])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_confusion[0, 0] + running_confusion[1, 1]) / dataset_sizes[phase]
            epoch_precision = running_confusion[0, 0] / (running_confusion[0, 0] + running_confusion[1, 0])
            epoch_recall = running_confusion[0, 0] / (running_confusion[0, 0] + running_confusion[0, 1])
            epoch_F1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)
            epoch_F2 = 5 * (epoch_precision * epoch_recall) / (4 * epoch_precision + epoch_recall)
            if phase == 'val':
                writer.add_scalar('Val/Loss', epoch_loss, epoch)
                writer.add_scalar('Val/Acc', epoch_acc, epoch)
                writer.add_scalar('Val/Precision', epoch_precision, epoch)
                writer.add_scalar('Val/Recall', epoch_recall, epoch)
                writer.add_scalar('Val/F1', epoch_F1, epoch)
                writer.add_scalar('Val/F2', epoch_F2, epoch)
                print("write val")
            else:
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Acc', epoch_acc, epoch)
                writer.add_scalar('Train/Precision', epoch_precision, epoch)
                writer.add_scalar('Train/Recall', epoch_recall, epoch)
                writer.add_scalar('Train/F1', epoch_F1, epoch)
                writer.add_scalar('Train/F2', epoch_F2, epoch)
                print("write train")

            print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F-1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_F1))
            print(running_confusion)
            # Keep the model with the best acc on val set (the evaluation metric could be changed)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Recall: {:4f}'.format(best_recall))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Loading the model
if args.model == 'pretrained':
    model_conv = torchvision.models.resnet18(pretrained=True)
elif args.model == 'age':
    model_conv = model_attention_lastest.resnet18(pretrained=False)
else:
    model_conv = model_attention_lastest.resnet18_attention(pretrained=False)
# for param in model_conv.parameters():
#     param.requires_grad = False

# Modify the output and input shape
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model_conv = model_conv.to(device)

# Set the loss function (with weight or without weight)
if args.weighted_loss:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.8]).to(device))
else:
    criterion = nn.CrossEntropyLoss()

# Set the optimizer config
# optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=0.001, momentum=0.9)
if args.model == 'pretrained':
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)
else:
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)

# Decay LR by a factor of 0.1 every 30 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=30, gamma=0.1)

# Train
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=epoches)
