from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

def main(max_iters, batch_size, learning_rate):
    device = torch.device('cpu')

    norm_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    model = convNet()

    train_loss = []
    train_acc = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    test_data = scipy.io.loadmat('../data/nist36_test.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    num_samples = train_x.shape[0]
    test_x, test_y = test_data['test_data'], test_data['test_labels']
    num_test_samples = test_x.shape[0]

    train_x = np.array([np.reshape(x, (32, 32)) for x in train_x])
    test_x = np.array([np.reshape(x, (32, 32)) for x in test_x])

    train_x_tensor = torch.from_numpy(train_x).type(torch.float32).unsqueeze(1)
    train_y_tensor = torch.from_numpy(train_y).type(torch.LongTensor)
    train_loader = DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)

    test_x_tensor = torch.from_numpy(test_x).type(torch.float32).unsqueeze(1)
    test_y_tensor = torch.from_numpy(test_y).type(torch.LongTensor)
    test_loader = DataLoader(TensorDataset(test_x_tensor, test_y_tensor), batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)

    for itr in range(max_iters):
        total_loss = 0
        correct = 0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs = torch.autograd.Variable(data[0])
            labels = torch.autograd.Variable(data[1])
            labels = torch.max(labels, 1)[1]

            # forward
            y_pred = model(inputs)
            loss = nn.functional.cross_entropy(y_pred, labels)

            # loss and accuracy
            total_loss += loss.item()
            predicted = torch.max(y_pred, 1)[1]
            correct += torch.sum(torch.eq(predicted, labels.data)).item()

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if itr % 10 == 0:
            print("itr: {:02d} \t total loss{:.2f}".format(i, total_loss))

        acc = correct/num_samples
        train_loss.append(total_loss)
        train_acc.append(acc)

    np.save("NIST36_conv_train_acc.npy", train_acc)
    np.save("NIST36_conv_train_loss.npy", train_loss)
    plt.figure('accuracy')
    plt.plot(range(max_iters), train_acc, color='r')
    plt.legend(['train accuracy'])
    plt.savefig('convNIST36acc.png')
    plt.show()

    plt.figure('loss')
    plt.plot(range(max_iters), train_loss, color='r')
    plt.legend(['train loss'])
    plt.savefig('convMNIST36.png')
    plt.show()

    print('Train accuracy: {}'.format(train_acc[-1]))

    torch.save(model.state_dict(), "q7_1_2_model_parameter.pkl")

    test_correctness = 0
    for data in test_loader:
        input = torch.autograd.Variable((data)[0])
        label = torch.autograd.Variable((data)[1])

        y_pred = model(input)
        loss = F.cross_entropy(y_pred, label)

        predicted = torch.max(y_pred, 1)[1]
        test_correctness += torch.sum(torch.eq(predicted, label.data)).item()

    test_acc = test_correctness/num_test_samples
    print('Test accuracy: {}'.format(test_acc))

if __name__ == '__main__':
    max_iters = 100
    batch_size = 64
    learning_rate = 0.01

    main(max_iters, batch_size, learning_rate)
