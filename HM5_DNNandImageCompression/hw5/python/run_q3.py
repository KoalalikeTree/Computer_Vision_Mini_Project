import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt



train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# for i in range(train_x.shape[0]):
#     if i >= 500:
#         visual = train_x[i, :].reshape((32, 32))
#         plt.imshow(visual, cmap='gray')
#         plt.show()
#         plt.imshow(visual.transpose(), cmap='gray')
#         plt.show()

max_iters = 100
# pick a batch size, learning rate
batch_size = 108
learning_rate = 3e-3
# learning_rate = 3e-4
# learning_rate = 3e-2
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, 64, params, name='layer1')
initialize_weights(64, 36, params, name='output')

losses = np.zeros((1, max_iters))
acces = np.zeros((1, max_iters))
valid_losses = np.zeros((1, max_iters))
valid_acces = np.zeros((1, max_iters))

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:

        # 正向传播
        layer1 = forward(xb, params, name='layer1', activation=sigmoid)
        prob = forward(layer1, params, name='output', activation=softmax)

        # 损失函数
        loss, acc = compute_loss_and_acc(yb, prob)
        # print(loss, acc)
        total_loss += loss
        total_acc += acc

        # 输出错误
        delta_output = prob - yb
        # 反向传播第一层
        delta_Layer1 = backwards(delta_output, params, name='output', activation_deriv=linear_deriv)
        # 反向传播第二层
        delta_input = backwards(delta_Layer1, params, name='layer1', activation_deriv=sigmoid_deriv)

        backwards(delta_Layer1, params, name='layer1', activation_deriv=sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['boutput'] -= learning_rate * params['grad_boutput']

    total_loss /= len(batches)
    losses[0, itr] = total_loss
    total_acc /= len(batches)
    acces[0, itr] = total_acc

    # run on validation set and report accuracy! should be above 75%
    valid_acc = None
    # 正向传播
    valid_layer1 = forward(valid_x, params, name='layer1', activation=sigmoid)
    valid_prob = forward(valid_layer1, params, name='output', activation=softmax)
    valid_losses[0, itr], valid_acces[0, itr] = compute_loss_and_acc(valid_y, valid_prob)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        print('Validation accuracy: ', valid_acces[0, itr], ' Validation loss', valid_losses[0, itr])

# Visualization
xlable = np.arange(max_iters).reshape(-1)
plt.figure(1)
ax = plt.gca()
ax.set_xlabel('ephoch')
ax.set_ylabel('loss')
ax.plot(xlable, losses.reshape(-1), 'g', label='training data', linewidth=0.5)
ax.plot(xlable, valid_losses.reshape(-1), 'r', label='validation data', linewidth=0.5)
title = 'learning rate = ' + str(learning_rate)
ax.set_title(title)
ax.legend()
filename = 'loss' + str(learning_rate) + '.png'
plt.savefig(filename)

plt.figure(2)
ax = plt.gca()
ax.set_xlabel('ephoch')
ax.set_ylabel('accuraccy')
ax.plot(xlable, acces.reshape(-1), 'g', label='training data', linewidth=0.5)
ax.plot(xlable, valid_acces.reshape(-1), 'r', label='validation data', linewidth=0.5)
title = 'learning rate = ' + str(learning_rate)
ax.set_title(title)
ax.legend()
filename = 'acc' + str(learning_rate) + '.png'
plt.savefig(filename)

plt.show()

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('q3_weights.pickle', 'rb') as handle:
   saved_params = pickle.load(handle)

# Learned weights
weights = saved_params['Wlayer1']

fig = plt.figure(3)
grid = ImageGrid(fig, 111, (8,8))
for i in range(64):
    weight_i = weights[:, i].reshape(32, 32)
    grid[i].imshow(weight_i)
filename = 'weight_trained.png'
plt.savefig(filename)
plt.show()

# Original weights
initialize_weights(1024, 64, saved_params, 'orig')
weights_orig = saved_params['Worig']

fig = plt.figure(4)
grid = ImageGrid(fig, 111, (8,8))
for i in range(64):
    weight_i = weights_orig[:, i].reshape(32, 32)
    grid[i].imshow(weight_i)
filename = 'weight_original.png'
plt.savefig(filename)
plt.show()


# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
# 测试集结果
test_layer1 = forward(valid_x, saved_params, 'layer1')
test_probs = forward(test_layer1, saved_params, 'output', softmax)

test_grotru = np.argmax(valid_y, axis=1)
test_predict = np.argmax(test_probs, axis=1)

num_example = valid_y.shape[0]
for i in range(num_example):
    confusion_matrix[test_grotru[i], test_predict[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
filename = 'confusion.png'
plt.savefig(filename)
plt.show()

print(confusion_matrix)
print(1)