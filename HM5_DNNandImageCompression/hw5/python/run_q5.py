import numpy as np
import scipy.io
from nn import *
from collections import Counter


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
losses = np.zeros((1, max_iters))

# Q5.1 & Q5.2
# initialize layers here

# get batches
batches = get_random_batches(train_x, train_x, batch_size)
num_batches = len(batches)

# weight initialization
initialize_weights(1024, 32, params, name='layer1')
initialize_weights(32, 32, params, name='layer2')
initialize_weights(32, 32, params, name='layer3')
initialize_weights(32, 1024, params, name='output')
Mw = Counter({'mW_layer1': 0, 'mW_layer2': 0, 'mW_layer3': 0, 'mW_output': 0,
              'mb_layer1': 0, 'mb_layer2': 0, 'mb_layer3': 0, 'mb_output': 0})

# train the weight
for itr in range(max_iters):
    total_acc = 0
    total_loss= 0
    acc=0

    for xb, _ in batches:
        # forward propagation
        layer1_node = forward(xb, params, name='layer1', activation=relu)
        layer2_node = forward(layer1_node, params, name='layer2', activation=relu)
        layer3_node = forward(layer2_node, params, name='layer3', activation=relu)
        output_layer = forward(layer3_node, params, name='output', activation=sigmoid)


        # backward propagation
        delta_output = output_layer - xb
        delta_layer3 = backwards(delta_output, params, name='output', activation_deriv=sigmoid_deriv)
        delta_layer2 = backwards(delta_layer3, params, name='layer3', activation_deriv=relu_deriv)
        delta_layer1 = backwards(delta_layer2, params, name='layer2', activation_deriv=relu_deriv)
        backwards(delta_layer1, params, name='layer1', activation_deriv=relu_deriv)

        Mw['mW_layer1'] = 0.9 * Mw['mW_layer1'] -learning_rate * params['grad_Wlayer1']
        Mw['mb_layer1'] = 0.9 * Mw['mb_layer1'] -learning_rate * params['grad_blayer1']
        params['Wlayer1'] += Mw['mW_layer1']
        params['blayer1'] += Mw['mb_layer1']

        Mw['mW_layer2'] = 0.9 * Mw['mW_layer2'] - learning_rate * params['grad_Wlayer2']
        Mw['mb_layer2'] = 0.9 * Mw['mb_layer2'] - learning_rate * params['grad_blayer2']
        params['Wlayer2'] += Mw['mW_layer2']
        params['blayer2'] += Mw['mb_layer2']

        Mw['mW_layer3'] = 0.9 * Mw['mW_layer3'] - learning_rate * params['grad_Wlayer3']
        Mw['mb_layer3'] = 0.9 * Mw['mb_layer3'] - learning_rate * params['grad_blayer3']
        params['Wlayer3'] += Mw['mW_layer3']
        params['blayer3'] += Mw['mb_layer3']

        Mw['mW_output'] = 0.9 * Mw['mW_output'] - learning_rate * params['grad_Woutput']
        Mw['mb_output'] = 0.9 * Mw['mb_output'] - learning_rate * params['grad_boutput']
        params['Woutput'] += Mw['mW_output']
        params['boutput'] += Mw['mb_output']

    layer1_node_valid = forward(valid_x, params, name='layer1', activation=relu)
    layer2_node_valid = forward(layer1_node_valid, params, name='layer2', activation=relu)
    layer3_node_valid = forward(layer2_node_valid, params, name='layer3', activation=relu)
    output_layer_valid = forward(layer3_node_valid, params, name='output', activation=sigmoid)

    loss = np.sum(np.square((output_layer_valid - valid_x)))
    losses[0, itr] = loss


    if itr % 10 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9



        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################
plt.figure(1)
ax = plt.gca()
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
epoches = np.arange(max_iters).reshape(max_iters, )
ax.plot(epoches, losses.reshape(max_iters, ), color='r', linewidth=.5, alpha=1.0, label='loss')
ax.legend()

plt.savefig('loss_auto.png')
plt.show()


num_valid_samples = valid_x.shape[0]
chosen_class = np.random.choice(np.arange(36), 5, False) * 100

valid_x = valid_data['valid_data']
layer1_node_valid = forward(valid_x, params, name='layer1', activation=relu)
layer2_node_valid = forward(layer1_node_valid, params, name='layer2', activation=relu)
layer3_node_valid = forward(layer2_node_valid, params, name='layer3', activation=relu)
output_layer_valid = forward(layer3_node_valid, params, name='output', activation=sigmoid)

for num_plot, num_class in enumerate(chosen_class):
    plt.subplot(5, 2, 2*num_plot+1)
    plt.imshow(valid_x[num_class].reshape(32, 32).T, cmap='gray')
    plt.subplot(5, 2, 2*num_plot+2)
    plt.imshow(output_layer_valid[num_class].reshape(32, 32).T, cmap='gray')
    plt.savefig('autoencoder.png')
plt.show()

# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################

psnr_total = 0

for num_sample in range(num_valid_samples):
    psnr_total += psnr(valid_x[num_sample], output_layer_valid[num_sample])
psnr_average = psnr_total / num_valid_samples

print("Average psnr", psnr_average)