from graphviz import Digraph, render
from IPython.display import display, Image, clear_output
import numpy as np
from sklearn import datasets
import time
import sys
import os

# parameters - data
args = sys.argv 
n = int(args[4])
n_outliers = 50
X, Y = datasets.make_regression(n_samples=n, n_features=2,
                                      n_informative=2, noise=10,
                                      coef=False, random_state=0)
Y = Y.reshape(n,1)
X = X + 1  
n_x = int(args[1])
n_h = int(args[2])
n_y = int(args[3])
lr = float(args[6])
iterations = int(args[5])


def Network(X, Y, n_x, n_h, n_y, iterations, lr):
    import numpy as np
    np.random.seed(1) 
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
        b1 = np.zeros(shape=(1, n_h))
        # Random initialization
        W2 = np.random.randn(n_y, n_h) * 0.01
        # bias
        b2 = np.zeros(shape=(1, n_y))
            
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
            
        return parameters 

    def sigmoid(Z):
        A = 1/(1+np.exp(-Z))
        return A    
    
    def sigmoid_derivate(Z):
      return sigmoid(Z)*(1-sigmoid(Z))       

    def cost_derivative(A,Y):
      return A - Y  

    def cost_function(X,Y):
      n = X.shape[0]
      return np.sum((X - Y)**2)/(2*n)                 
    
    def gradient_descent_update(parameters, derivatives_w, derivatives_b, n, lr):
        # updating parameters with gradient descent
        parameters['W1'] = parameters['W1'] - (derivatives_w[0] * lr) 
        parameters['W2'] = parameters['W2'] - (derivatives_w[1] * lr)
        parameters['b1'] = parameters['b1'] - (derivatives_b[0] * lr)
        parameters['b2'] = parameters['b2'] - (derivatives_b[1] * lr)
        return parameters

    def backprop(X, Y, parameters=None):
        def linear_forward(A, W, b):
            # neuron computing for a layer
            Z = np.dot(A, W.T) + b
            
            return Z

        def linear_activation_forward(A_prev, W, b, activation):
            """
            Implement the forward propagation for the LINEAR->ACTIVATION layer

            Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (number of neurons in the current layer, number of neurons in the previous layer)
            b -- bias vector, numpy array of shape (number of neurons in the current layer, 1)

            Returns:
            A -- the output of the activation function, also called the post-activation value 
            Z -- the input of the activation function, also called pre-activation parameter 
            """
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z = linear_forward(A_prev, W, b)
            
            if activation=='No activation':
                # return the same value
                A = Z
            else:
                # compute sigmoid transformation
                A = sigmoid(Z)

            return A, Z     
        
        A = X
        As = [X]
        Zs = []

        # forward step across entire network
        # for Hidden network
        A, Z = linear_activation_forward(A, parameters['W1'], parameters['b1'], 'sigmoid')
        # 
        As.append(A)
        Zs.append(Z)        

        # for last layer L - ouput layer          
        A, Z = linear_activation_forward(A, parameters['W2'], parameters['b2'], 'No activation')
        As.append(A)
        Zs.append(Z)
        
        # backward step   
        # for last layer L     
        error_L = cost_derivative(As[-1], Y) * Zs[-1]
        cost = cost_function(As[-1], Y)
        derivatives_b[-1] = np.sum(error_L, axis=0, keepdims=True)/n
        derivatives_w[-1] = (1/n)*np.dot(error_L.T, As[-2])
        
        # for hidden layer
        error_hidden_l = np.dot(error_L, parameters['W2']) * sigmoid_derivate(Zs[-2])
        derivatives_b[-2] = np.sum(error_hidden_l, axis=0, keepdims=True)/n
        derivatives_w[-2] = (1/n)*np.dot(error_hidden_l.T, As[-3])

        # plot graph
        net = Digraph('net', filename='network.gv')
        net.attr(rankdir='LR', ranksep='1.4', splines='true', labelloc='t', label='Neural Network example')


        with net.subgraph(name='cluster_1') as c:
          c.node(str(round(np.sum(As[1],axis=0)[0],3)))
          c.node(str(round(np.sum(As[1],axis=0)[1],3)))
          c.node(str(round(np.sum(As[1],axis=0)[2],3)))
          c.attr(label='Layer 2 (Hidden layer)')

        with net.subgraph(name='cluster_0') as c:
          c.node('Input 1')
          c.node('Input 2')
          c.attr(label='Layer 1 (Input layer)')  

        with net.subgraph(name='cluster2') as c:
          c.node(str(round(np.sum(As[-1],axis=0)[0],3)))
          c.attr(label='Layer 3 (Output layer)')    

        net.node(str(round(cost,3)), shape='box')  
        net.edge(str(round(np.sum(As[-1],axis=0)[0],3)), str(round(cost,3)))      
        net.node('Input 1', shape='box')
        net.node('Input 2', shape='box')
        net.edge('Input 1', str(round(np.sum(As[1],axis=0)[0],3)), label=str(round(parameters["W1"][1,0],3)))
        net.edge('Input 1', str(round(np.sum(As[1],axis=0)[0],3)), label=str(round(parameters["W1"][0,0],3)))
        net.edge('Input 1', str(round(np.sum(As[1],axis=0)[2],3)), label=str(round(parameters["W1"][2,0],3)))
        net.edge('Input 2', str(round(np.sum(As[1],axis=0)[1],3)), label=str(round(parameters["W1"][0,1],3)))
        net.edge('Input 2', str(round(np.sum(As[1],axis=0)[1],3)), label=str(round(parameters["W1"][1,1],3)))
        net.edge('Input 2', str(round(np.sum(As[1],axis=0)[2],3)), label=str(round(parameters["W1"][2,1],3)))
        net.edge(str(round(np.sum(As[1],axis=0)[0],3)), str(round(np.sum(As[-1],axis=0)[0],3)), label=str(round(parameters["W2"][0,0],3)))
        net.edge(str(round(np.sum(As[1],axis=0)[1],3)), str(round(np.sum(As[-1],axis=0)[0],3)), label=str(round(parameters["W2"][0,1],3)))
        net.edge(str(round(np.sum(As[1],axis=0)[2],3)), str(round(np.sum(As[-1],axis=0)[0],3)), label=str(round(parameters["W2"][0,2],3)))
        net.view()
        render('dot', 'png', 'network.gv') 
        display(Image(filename='network.gv.png')) 
        clear_output(wait=True)
        time.sleep(0.2)
        os.remove("network.gv.png") 
        os.remove("network.gv")
        os.remove("network.gv.pdf")        


        return (derivatives_w, derivatives_b, cost, np.sum(error_L))
    
    # run network
    # number of samples 
    n = X.shape[0]
    costs = []
    losses = []
    for iteration in range(iterations):
        if iteration == 0:   
            # initialization
            # parameters, weights and bias for hidden and output layer
            parameters = initialize_parameters(n_x, n_h, n_y)
        # initial gradients of parameters
        derivatives_b = [np.zeros(b.shape) for b in [parameters['b1'], parameters['b2']]]
        derivatives_w = [np.zeros(w.shape) for w in [parameters['W1'], parameters['W2']]]
        # computing the backprop algorithm for the entire data X
        # calcute partial derivatives
        derivatives_w, derivatives_b, cost, error = backprop(X, Y, parameters=parameters)
        costs.append(cost)
        losses.append(error)
        # run gradient descent for update parameters
        parameters = gradient_descent_update(parameters, derivatives_w, derivatives_b, n, lr)

    return parameters, costs, losses   

parameters, costs, losses = Network(X, Y, n_x, n_h, n_y, iterations, lr)  
