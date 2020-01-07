import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    low = np.negative(np.sqrt(6/(in_size + out_size)))
    high = np.negative(low)

    # If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn
    W = np.random.uniform(low, high, (in_size, out_size))  # dimension: in_size * out_size
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    res = np.divide(1, 1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    # X dimension: Examples * input_layer
    W = params['W' + name]   # Dimension: input_layer * output layer
    b = params['b' + name]   # Dimension: output_layer * 1

    # [Examples * D_input] [D_input * D_output] + D_output
    pre_act = np.dot(X, W) + b   # Dimension: Examples * output_layers
    post_act = activation(pre_act)  # Dimension: Examples * output_layers

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    c = np.max(x, axis=1).reshape(-1, 1)  # Dimension: examples * 1

    deno = np.exp(x - c)  # Dimension: examples * classes
    nume = np.sum(deno, axis = 1).reshape(-1, 1)  # Dimension: examples * 1

    res = np.divide(deno, nume)  # Dimension: examples * 1

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    num_examples = y.shape[0]

    loss = -np.sum(np.multiply(y, np.log(probs)))

    groutru_label = np.argmax(y, axis=-1)  # Dimension: examples * D_ourtput
    predict_label = np.argmax(probs, axis=-1)  # Dimension: examples * D_output
    correct_label = np.equal(groutru_label, predict_label)  # Dimension: examples * D_output
    acc = np.sum(correct_label) / num_examples

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res


def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop y0-y
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]  # Dimension: input_layer * output_layer
    b = params['b' + name]  # Dimension: output_layer * 1
    X, pre_act, post_act = params['cache_' + name]
    num_examples = X.shape[0]
    # X Dimension: E * input
    # post_act: E * output
    # pre_act Dimension: E * output
    # delta : E * output
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    delta = delta * activation_deriv(post_act)  # Dimension: E * output
    grad_W = np.dot(X.T, delta)
    grad_X = np.dot(delta, W.T)
    grad_b = np.dot(np.ones((1, delta.shape[0])), delta).reshape(-1)

    # grad_W = np.dot(X.reshape(-1, num_examples), delta.reshape(num_examples, -1))  # Dimension: input * output
    # assert grad_W.shape == W.shape, "The shape of grad_W is not correct"
    #
    # grad_X = np.dot(delta.reshape(num_examples, -1), np.transpose(W))  # Dimension: E * input
    # assert grad_X.shape == X.shape, "The shape of grad_X is not correct"
    #
    # grad_b = np.dot(np.ones((1, num_examples)), delta).reshape(-1)   # Dimension: 1 * output
    # assert grad_b.shape == b.shape, "The shape of grad_b is not correct"

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []

    num_inp_examples = x.shape[0]
    num_inp_dim = x.shape[1]
    num_outp_dim = y.shape[1]
    num_batches = np.int(num_inp_examples / batch_size)
    # batch_size could be 50 for 200 examples

    for batch_num in range(num_batches):
        choosen_id = np.random.choice(num_inp_examples, batch_size, False)
        choosen_x = np.zeros((batch_size, num_inp_dim))
        choosen_x[:, :] = x[choosen_id, :]

        choosen_y = np.zeros((batch_size, num_outp_dim))
        choosen_y[:, :] = y[choosen_id, :]

        batches.append((choosen_x, choosen_y))

    return batches
