import numpy as np

class fcnn:
    def __init__(self,
                 node_layers = [2, 3, 3],
                 n_hidden_layers = 1,
                 learning_rate = 0.25,
                 tol = 1e-3,
                 max_epoch = 1000, 
                 output_activation = 'sigmoid'):
        '''
        Parameters:
        node_layer: Number of nodes per layer, input layer + all hidden + output layer
        n_hidden_layer: Number of hidden layers in the network
        learning_rate: Constant learning rate
        tol: Tolerance of error between consequtive epochs
        max_epoch: Max number of epochs for training
        output_activation: To define the output activation function

        Returns:
        None
        '''
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_epoch = max_epoch
        self.output_activation = output_activation
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
        if self.output_activation == "sigmoid":
            ho = self.logistic_sigmoid(ao) # output layer signal
        elif self.output_activation == "linear":
            ho = ao

        return ho, ao, h1, a1

    def backward_propagate(self, ho, h1, label, sample):
        x_hat = np.concatenate(([1], sample)) # augmenting the input vector
        if self.output_activation == "sigmoid":
            delta_o = ((label-ho)*ho*(1-ho))
        elif self.output_activation == "linear":
            delta_o = (label-ho)
        self.Wo += self.learning_rate*np.outer(h1, delta_o)

        delta_h = np.matmul(self.Wo, delta_o)*h1*(1-h1)
        self.Wh += self.learning_rate*np.outer(x_hat, delta_h)
    
    def fit(self, datas, labels):
        '''
        This method fits the model to the required data
        '''
        err_epoch = []
        # run epochs; stopping criteria 1
        for epoch in range(self.max_epoch):
            err = [] # store instantaneous error per example
            for label, sample in zip(labels, datas): # show label and data of each class
                # forward propagate through the network
                ho, _, h1, _ = self.forward_propagate(sample)
                err.append(self.inst_err(label, ho)) # store instantenous error of each example

                # update processs
                self.backward_propagate(ho, h1, label, sample)
            err_epoch.append(np.average(err)) # store average error per epoch
        return err_epoch
        
    def fit_classifier(self, class1, class2, class3):
        '''
        This methods fits the model for classification
        '''
        labels = { 0: [[1, 0, 0]],
                    1: [[0, 1, 0]],
                    2: [[0, 0, 1]]}
        
        all_data = np.concatenate((class1, class2, class3), axis=0)
        all_true_labels = np.concatenate((labels[0]* class1.shape[0],
                                          labels[1]* class2.shape[0],
                                          labels[2]* class3.shape[0]), axis=0)
        return self.fit(all_data, all_true_labels)
    
    def fit_regressor(self, data):
        '''
        This model fits the model for regression
        '''
        target = data[:, -1]
        features = data[:, :-1].reshape((data.shape[0], data.shape[1]-1)) # for the sake of uniformity
        return self.fit(features, target)
    
    def predict(self, sample):
        ho, _, _, _ = self.forward_propagate(sample)
        return np.argmax(ho) + 1
    
    def regress(self, sample):
        ho, _, _, _ = self.forward_propagate(sample)
        return ho

    
