from graphviz import Digraph, render
from IPython.display import display, Image, clear_output
import numpy as np
import time
import sys
import os

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- number of neurons in the input layer = 2
    n_h -- number of neurons in the hidden layer = 3
    n_y -- number of neurons in the output layer = 1
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    # Random initialization
    W1 = np.random.randn(n_h, n_x) * 0.01
    # bias
    b1 = np.zeros(shape=(n_h, 1))
    # Random initialization
    W2 = np.random.randn(n_y, n_h) * 0.01
    # bias
    b2 = np.zeros(shape=(n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

args = sys.argv 
parameters = initialize_parameters(int(args[1]),int(args[2]),int(args[3])) 

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (number of neurons in the current layer, number of neurons in the previous layer)
    b -- bias vector, numpy array of shape (number of neurons in the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    # neuron computing for a layer
    Z = np.dot(W, A) + b
    # importan information for backward step
    cache = (A, W, b)
    
    return Z, cache


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache



def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (number of neurons in the current layer, number of neurons in the previous layer)
    b -- bias vector, numpy array of shape (number of neurons in the current layer, 1)

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation=='No activation':
        identity = lambda z: (z, z) # return the same value
        A, activation_cache = identity(Z)
    else:
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache   


# testing - This example has only been validated for three layers similar to the figure

# for layer 1 - We will consider a input matrix with 2 samples of 1 dimension
inputs = np.random.randint(low=0, high=255 ,size=(2,1), dtype=np.uint8)
A0 = inputs
W1 = parameters['W1']
# create the bias vector
b1 = np.zeros((3,1))
A1, linear_activation_forward_1 = linear_activation_forward(A0, W1, b1, activation='sigmoid')

# for layer 2 - Receive the above output 
W2 = parameters['W2']
b2 = np.zeros((1,1))
A2, linear_activation_forward_2 = linear_activation_forward(A1, W2, b2, activation='Not activation')

# plot 
net = Digraph('net', filename='network.gv')
net.attr(rankdir='LR', ranksep='1.4', splines='true', labelloc='t', label='Forward propagation + cost')

with net.subgraph(name='cluster_1') as c:
  c.node(str(round(A1[0,0],3)))
  c.node(str(round(A1[1,0],3)))
  c.node(str(round(A1[2,0],3)))
  c.attr(label='Layer 2 (Hidden layer)')

with net.subgraph(name='cluster_0') as c:
  c.node(str(round(A0[0,0],3)))
  c.node(str(round(A0[1,0],3)))
  c.attr(label='Layer 1 (Input layer)')  

with net.subgraph(name='cluster2') as c:
  c.node(str(round(A2[0,0],3)))
  c.attr(label='Layer 3 (Output layer)')    

net.node('cost=(20-'+str(round(A2[0,0],3))+')²', shape='box')    
net.node(str(round(A0[0,0],3)), shape='box')
net.node(str(round(A0[1,0],3)), shape='box')
net.edge(str(round(A2[0,0],3)), 'cost=(20-'+str(round(A2[0,0],3))+')²', shape='box')
net.edge(str(round(A0[0,0],3)), str(round(A1[1,0],3)), label=str(round(parameters["W1"][1,0],3)))
net.edge(str(round(A0[0,0],3)), str(round(A1[0,0],3)), label=str(round(parameters["W1"][0,0],3)))
net.edge(str(round(A0[0,0],3)), str(round(A1[2,0],3)), label=str(round(parameters["W1"][2,0],3)))
net.edge(str(round(A0[1,0],3)), str(round(A1[0,0],3)), label=str(round(parameters["W1"][0,1],3)))
net.edge(str(round(A0[1,0],3)), str(round(A1[1,0],3)), label=str(round(parameters["W1"][1,1],3)))
net.edge(str(round(A0[1,0],3)), str(round(A1[2,0],3)), label=str(round(parameters["W1"][2,1],3)))
net.edge(str(round(A1[0,0],3)), str(round(A2[0,0],3)), label=str(round(parameters["W2"][0,0],3)))
net.edge(str(round(A1[1,0],3)), str(round(A2[0,0],3)), label=str(round(parameters["W2"][0,1],3)))
net.edge(str(round(A1[2,0],3)), str(round(A2[0,0],3)), label=str(round(parameters["W2"][0,2],3)))
net.view()
render('dot', 'png', 'network.gv') 
display(Image(filename='network.gv.png')) 

os.remove("network.gv.png") 
os.remove("network.gv")
os.remove("network.gv.pdf")