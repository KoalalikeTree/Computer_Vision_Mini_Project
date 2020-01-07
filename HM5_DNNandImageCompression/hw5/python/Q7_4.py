import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import os
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches

device = torch.device('cpu')
batch_size=64
max_iters = 50
learning_rate = 0.01

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

trainset = datasets.EMNIST('./data', split='balanced', train=True,
                                       download=True, transform = transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers=0)

testset = torchvision.datasets.EMNIST('./data', split='balanced', train=False,
                                     download=True, transform= transforms)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

class lenet5(nn.Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=(5, 5)),
                                   nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(120, 84),
                                 nn.ReLU(),
                                 nn.Linear(84, 47),
                                 nn.LogSoftmax(dim=-1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 120)
        x = self.fc1(x)
        return x

model = lenet5()
trainLoss = []
trainAcc = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for itr in range(max_iters):
    totalLoss=0
    correctness=0
    for data in trainloader:
        inputsLayer = torch.autograd.Variable(data[0])
        grouthLabers = torch.autograd.Variable(data[1])

        predictedLabel = model(inputsLayer)
        loss = nn.functional.cross_entropy(predictedLabel, grouthLabers)

        totalLoss += loss.item()
        predicted = torch.max(predictedLabel, 1)[1]
        correctness += torch.sum(torch.eq(predicted, grouthLabers)).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correctness/len(trainset)
    trainLoss.append(totalLoss)
    trainAcc.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, trainLoss, acc))

plt.figure('accuracy')
plt.plot(range(max_iters), trainAcc, color='r')
plt.legend(['train accuracy'])
plt.savefig('pytorch_EMNIST_train_acc.png')
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), trainLoss, color='r')
plt.legend(['train loss'])
plt.savefig('pytorch_EMNIST_train_acc.png')
plt.show()

print('Train accuracy: {}'.format(trainAcc[-1]))

torch.save(model.state_dict(), "q7_1_4_model_parameter.pkl")

test_correct = 0
for data in test_loader:
    # get the inputs
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])

    # get output
    y_pred = model(inputs)
    loss = nn.functional.cross_entropy(y_pred, labels)

    predicted = torch.max(y_pred, 1)[1]
    test_correct += torch.sum(predicted == labels.data).item()

test_acc = test_correct/len(testset)

print('Test accuracy: {}'.format(test_acc))



