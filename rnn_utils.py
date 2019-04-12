import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, curr_loss):
    return loss * 0.999 + curr_loss * 0.001

def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values
    
    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.1 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.1 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.1 # hidden to hidden
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # hidden bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by":by }
    
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, b = parameters['Waa'], parameters['Wax'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state

    return a_next

def rnn_step_backward(gradients, parameters, x, a, a_prev):
    
    da = gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity

    #print('da:', da)
    #print('daraw:', daraw)
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['b']  += -lr * gradients['db']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['by']  += -lr * gradients['dby']
    
    return parameters

def rnn_forward(X, Y, a0, parameters, width=28, height=28):
    
    # Initialize x, a as empty dictionaries
    x, a = {}, {}
    
    a[-1] = np.copy(a0)
    
    X = X.reshape((height, width))
    time_steps = 1
    step_rows = 28
    n_x = int(width * step_rows)
    
    # initialize your loss to 0
    loss = 0
    
    for t in range(time_steps):

        #x[t] = X[step_rows*t: step_rows*(t+1) , :]
        x[t] = X.reshape((n_x, 1))
        #print("Timestep: {}, x[{}] is:".format(t, t), x[t])
        
        # Run one step forward of the RNN
        a[t] = rnn_step_forward(parameters, a[t-1], x[t])
        #print("Timestep: {}, a[{}] is:".format(t, t), a[t])
        
    y_hat = softmax(np.dot(parameters['Wya'], a[-1]) + parameters['by']) # unnormalized log probabilities
    #print( "y_hat:", y_hat)
    #print( "Predicted Value:", np.argmax(y_hat) )
        
    # Update the loss by substracting the cross-entropy term from it.
    loss -= np.log(y_hat[Y])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_backward(X, Y, parameters, cache, time_steps=1):
    
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    dy = np.copy(y_hat)
    dy[Y] -= 1
    
    #print( "dy:", dy)

    gradients['dWya'] = np.dot(dy, a[-1].T)
    gradients['dby'] = dy
    gradients['da_next'] = np.dot(parameters['Wya'].T, dy)
    #print("Wya:", parameters['Wya'].T)
    #print("da_next:", gradients['da_next'])
    
    # Backpropagate through time
    for t in reversed(range(time_steps)):
        gradients = rnn_step_backward(gradients, parameters, x[t], a[t], a[t-1])
    ### END CODE HERE ###
    
    return gradients, a

