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
net = Digraph('net', filename='network.gv')
net.attr(rankdir='LR', ranksep='1.4', splines='true', labelloc='t', label='Network initialization')

with net.subgraph(name='cluster_1') as c:
  c.node('Neuron 1')
  c.node('Neuron 2')
  c.node('Neuron 3')
  c.attr(label='Layer 2 (Hidden layer)')

with net.subgraph(name='cluster_0') as c:
  c.node('Input 1')
  c.node('Input 2')
  c.attr(label='Layer 1 (Input layer)')  

with net.subgraph(name='cluster2') as c:
  c.node('Output 1')
  c.attr(label='Layer 3 (Output layer)')    
    
net.node('Input 1', shape='box')
net.node('Input 2', shape='box')
net.edge('Input 1', 'Neuron 2', label=str(round(parameters["W1"][1,0],3)))
net.edge('Input 1', 'Neuron 1', label=str(round(parameters["W1"][0,0],3)))
net.edge('Input 1', 'Neuron 3', label=str(round(parameters["W1"][2,0],3)))
net.edge('Input 2', 'Neuron 1', label=str(round(parameters["W1"][0,1],3)))
net.edge('Input 2', 'Neuron 2', label=str(round(parameters["W1"][1,1],3)))
net.edge('Input 2', 'Neuron 3', label=str(round(parameters["W1"][2,1],3)))
net.edge('Neuron 1', 'Output 1', label=str(round(parameters["W2"][0,0],3)))
net.edge('Neuron 2', 'Output 1', label=str(round(parameters["W2"][0,1],3)))
net.edge('Neuron 3', 'Output 1', label=str(round(parameters["W2"][0,2],3)))
net.view()
render('dot', 'png', 'network.gv') 
display(Image(filename='network.gv.png')) 

os.remove("network.gv.png") 
os.remove("network.gv")
os.remove("network.gv.pdf")