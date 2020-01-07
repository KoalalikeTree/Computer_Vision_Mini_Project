import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

device = torch.device('cpu')

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

input_dim = train_x.shape[1]
output_dim = train_y.shape[1]
sample_num = train_x.shape[0]
test_sample_num = test_x.shape[0]

max_iters = 200
batch_size = 64
learning_rate = 0.01
hidden_size = 64

train_x_tensor = torch.from_numpy(train_x).type(torch.float32)
# print(train_x_tensor.size())
# print(train_x_tensor)
train_y_tensor = torch.from_numpy(train_y).type(torch.LongTensor)
# print(train_y_tensor.size())
# print(train_y_tensor)
train_loader = DataLoader(TensorDataset(train_x_tensor,train_y_tensor), batch_size=batch_size, shuffle=True)

test_x_tensor = torch.from_numpy(test_x).type(torch.float32)
test_y_tensor = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(test_x_tensor, test_y_tensor), batch_size=batch_size, shuffle=True)

class fully_nn(nn.Module):
    def __init__(self, input_dim, hidden_node_num, output_dim):
        super(fully_nn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_node_num)
        self.fc2 = nn.Linear(hidden_node_num, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model = fully_nn(input_dim, hidden_size, output_dim)

running_loss = 0.0
train_acc = []
train_loss = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
for epoch in range(max_iters):
    total_loss = 0
    total_acc = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        # equals to argmax
        ground_tru = torch.max(labels, 1)[1]

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(inputs)
        prediction = torch.max(output, 1)[1]

        total_acc += torch.sum(prediction.eq(ground_tru))

        loss = criterion(output, ground_tru)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    acc = total_acc.numpy() / sample_num
    train_loss.append(total_loss)
    train_acc.append(acc)

    if epoch % 10 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch, total_loss, acc))

# visualize the loss and accuracy
plt.figure('accuracy')
plt.plot(range(max_iters), train_acc, color='r')
plt.legend(['train accuracy'])
plt.savefig('pytorch_fcnn_train_acc.png')
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='r')
plt.legend(['train loss'])
plt.savefig('pytorch_fcnn_train_loss.png')
plt.show()

PATH = './q7_1_net.pth'
torch.save(model.state_dict(), PATH)
print('Finished Training')

total_acc_test = 0.0
for data in test_loader:
    inputs, labels = data

    # equals to argmax
    ground_tru = torch.max(labels, 1)[1]

    # zero the parameter gradients
    optimizer.zero_grad()

    output = model(inputs)
    prediction = torch.max(output, 1)[1]

    total_acc_test += torch.sum(prediction.eq(ground_tru))

test_acc = total_acc_test.numpy() / test_sample_num
print('Test accuracy: {}'.format(test_acc))

print('Finished Testing')

