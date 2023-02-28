import numpy as np

class fcnn:
    def __init__(self,
                 node_layers = [2, 3, 3],
                 n_hidden_layers = 1,
                 learning_rate = 0.25,
                 tol = 1e-3,
                 max_epoch = 1000):
        '''
        Parameters:
        node_layer: Number of nodes per layer, input layer + all hidden + output layer
        learning_rate: Constant learning rate
        tol: Tolerance of error between consequtive epochs
        max_epoch: Max number of epochs for training

        Returns:
        None
        '''
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_epoch = max_epoch

        self.input_dim = node_layers[0]
        self.nodes_h_dim = node_layers[1]
        self.output_dim = node_layers[-1]
        # initializing weights
        self.Wh = np.random.randn(self.input_dim+1, self.nodes_h_dim) # a = W.T @ x; rows of Wh is the dimention of input
        self.Wo = np.random.randn(self.nodes_h_dim, self.output_dim) # a = W.T @ x; rows of Wh is the dimention of input
        return None

    def logistic_sigmoid(self, x):
        '''
        Logistic sigmoidal function
        '''
        return 1/(1+np.exp(-x))   

    def inst_err(self, y, y_hat):
        '''
        Return the instantaneous error for each example
        '''
        return 0.5*(y-y_hat)**2  
      
    def forward_propagate(self, x):
        '''
        Computer the output of the network for a given example propagating forward in the network

        Parameter:
        Sample input vector

        Return:
        ho, h1 (returns in reverse order of operation)
        '''
        # input layer
        x_hat = np.concatenate(([1], x)) # augmenting the input vector
        # hidden layer
        a1 = np.matmul(self.Wh.T, x_hat) # hidden layer activation value
        h1 = self.logistic_sigmoid(a1) # hidden layer signal
        #output layer
        ao = np.matmul(self.Wo.T, h1) # output layer activation value
        ho = self.logistic_sigmoid(ao) # output layer signal

        return ho, h1
        # return ao, ho, a1, h1

    def backward_propagate(self, ho, h1, label, sample):
        # delta_o = np.outer((label - ao), )
        x_hat = np.concatenate(([1], sample)) # augmenting the input vector

        delta_o = ((label-ho)*ho*(1-ho))
        self.Wo += self.learning_rate*np.outer(h1, delta_o)

        delta_h = np.matmul(self.Wo, delta_o)*h1*(1-h1)
        self.Wh += self.learning_rate*np.outer(x_hat, delta_h)
    
    def fit(self, class1, class2, class3):
        labels = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        datas = [class1, class2, class3]
        err_epoch = []

        # run epochs; stopping criteria 1
        for epoch in range(self.max_epoch):
            err = [] # store instantaneous error per example
            for label, data in zip(labels, datas): # show label and data of each class
                for sample in data:
                    # forward propagate through the network
                    ho, h1 = self.forward_propagate(sample)
                    err.append(self.inst_err(label, ho)) # store instantenous error of each example

                    # update processs
                    self.backward_propagate(ho, h1, label, sample)
            err_epoch.append(np.average(err)) # store average error per epoch
        return err_epoch

    def predict(self, sample):
        ho, _ = self.forward_propagate(sample)
        return np.argmax(ho) + 1
    
# import numpy as np

# class fcnn:
#     def __init__(self, node_layers = [], learning_rate = 0.25, tol = 1e-3, max_epoch = 1000):
#         '''
#         Parameters:
#         node_layer: Number of nodes per layer, input layer + all hidden + output layer
#         learning_rate: Constant learning rate
#         tol: Tolerance of error between consequtive epochs
#         max_epoch: Max number of epochs for training

#         Returns:
#         None
#         '''
#         self.weights = {} # dictionary to track weights of each layer
#         self.learning_rate = learning_rate
#         self.tol = tol
#         self.max_epoch = max_epoch
#         self.node_layers = node_layers


#         pass
#     def init_weights(self):
#         '''
#         Initialise weights beween each layer
#         '''
#         for i, _ in enumerate(self.node_layers[:-1], start=1):
#             self.layers[i] = np.random.randn(self.node_layers[i]+1, self.node_layers[i+1])

#     def logistic_sigmoid(self, x):
#         '''
#         Logistic sigmoid function
#         '''
#         return 1/(1+np.exp(-x))
#     def forward_propagation(self):
#         pass
#     def backward_propagation(self):
#         pass
#     def fit(self):
#         pass
#     def predict(self):
#         pass