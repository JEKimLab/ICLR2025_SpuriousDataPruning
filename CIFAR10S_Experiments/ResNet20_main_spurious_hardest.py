import argparse
import numpy as np
import pandas as pd
import os
import shutil
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
from collections import defaultdict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser(description="get core type.")
parser.add_argument('core_type', type=str, default="hardest")
args = parser.parse_args()

trans_first = transforms.PILToTensor()
train_dataset = datasets.CIFAR10('./data.cifar10', train=True, download=True, transform=trans_first)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = False)

test_dataset = datasets.CIFAR10('./data.cifar10', train=False, download=True, transform=trans_first)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False)

trans_train = transforms.Compose([transforms.Pad(4), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trans_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trans1 = transforms.ToTensor()
trans4 = transforms.RandomCrop(32)
trans3 = transforms.Pad(4)
trans5 = transforms.RandomHorizontalFlip()
trans6 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transinit = transforms.ToPILImage()

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, labels, is_spurious, indexs, transform=None):
        self.image_list = image_list
        self.labels = labels
        self.is_spurious = is_spurious
        self.indexs = indexs
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]
        spurious = self.is_spurious[idx]
        index = self.indexs[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, spurious, index


class CustomImageDatasetTest(torch.utils.data.Dataset):
    def __init__(self, image_list, labels, transform=None):
        self.image_list = image_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

image_index_pairs = []

with open('all_thirds.pkl', 'rb') as f:
    x1, x2, x3 = pickle.load(f)

if args.core_type == "hardest":
    to_consider = x3
else:
    to_consider = x1

def create_spurious_train(trainloader, sc_label = 0):
  new_images = []
  new_labels = []
  new_spurious = []
  new_index = []
  index_counter = 0
  for i, data in enumerate(trainloader):
    image, label = data[0][0], data[1][0]
    spurious = 0
    if label.item() == sc_label and index_counter in to_consider:
      image[0][:, 16] = (0.75) * 255
      spurious += 1
    new_images.append(transinit(image))
    new_labels.append(label)
    new_spurious.append(spurious)
    new_index.append(index_counter)
    image_index_pairs.append([transinit(image), index_counter])
    index_counter += 1
  new_dataset = CustomImageDataset(image_list=new_images, labels=new_labels, is_spurious=new_spurious, indexs=new_index, transform=trans_train)

  print('index_counter', index_counter)
  train_loader = torch.utils.data.DataLoader(new_dataset, shuffle = True, batch_size = 64)

  return train_loader

train_loader = create_spurious_train(train_loader, 0)

def create_spurious_test(testloader, sc_label = 1):
  new_images = []
  new_labels = []
  for i, data in enumerate(testloader):
    image, label = data[0][0], data[1][0]
    if label.item() == sc_label:
      image[0][:, 16] = (0.75) * 255
    new_images.append(transinit(image))
    new_labels.append(label)
  
  new_dataset = CustomImageDatasetTest(image_list=new_images, labels=new_labels, transform=trans_test)
  test_loader = torch.utils.data.DataLoader(new_dataset, shuffle = False, batch_size = 64)

  return test_loader

test_loader = create_spurious_test(test_loader, 1)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=54 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

training_required = True
if training_required:

  model = resnet(depth = 20).to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
  criterion_CE = nn.CrossEntropyLoss()

  epochs = 160

  checkpoint_every = 170

  best_test = 0.

  for epoch in range(1, epochs + 1, 1):
      model.train()
      avg_loss = 0.
      training_acc = 0.
      testing_acc = 0.
      if epoch == 80 or epoch == 120:
          for param_group in optimizer.param_groups:
              param_group['lr'] *= 0.1

      for batch_idx, (images, labels, is_spurious, index) in enumerate(train_loader):
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()

          outputs = model(images)
          _, predictions = torch.max(outputs.data, 1)
          training_acc += (predictions == labels).sum().item()

          loss = criterion_CE(outputs, labels)
          avg_loss += loss.item()
          loss.backward()

          optimizer.step()

      print(f'Epoch {epoch} -> Loss: {avg_loss/len(train_loader)}, Training Accuracy: {training_acc/len(train_loader.dataset)}', end = ' ')

      if epoch % checkpoint_every == 0 and epoch != 0:
          torch.save(model, 'ResNet_epoch' + epoch + '.pt')

      model.eval()
      for batch_idx, (test_images, test_labels) in enumerate(test_loader):
          test_images, test_labels = test_images.to(device), test_labels.to(device)

          test_outputs = model(test_images)
          _, test_predictions = torch.max(test_outputs.data, 1)
          testing_acc += (test_predictions == test_labels).sum().item()

      print('Testing Accuracy: ', testing_acc/len(test_loader.dataset))

  torch.save(model, 'ResNet20_best_hardest.pt')
  print('Test Accuracy:', end = ' ')

  to_test = torch.load('ResNet20_best_hardest.pt')

  testing_acc = 0.
  d = defaultdict(int)

  to_test.eval()
  for batch_idx, (test_images, test_labels) in enumerate(test_loader):
      test_images, test_labels = test_images.to(device), test_labels.to(device)

      test_outputs = to_test(test_images)
      _, test_predictions = torch.max(test_outputs.data, 1)
      for j in range(len(test_predictions)):
          if test_labels[j].item() == 1:
              if test_predictions[j].item() == 1:
                  pass 
              else:
                  d[test_predictions[j].item()] += 1
  print('Spurious Misclassifications:', d[0])
