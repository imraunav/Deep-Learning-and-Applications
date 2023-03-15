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
    
    def fit(self, datas, labels, valid_data, valid_labels):
        '''
        This method fits the model to the required data
        '''
        err_epoch = []
        valid_loss_epoch = []
        # run epochs; stopping criteria 1
        # valid_pred_labels = []
        for epoch in range(self.max_epoch):
            err = [] # store instantaneous error per example
            for label, sample in zip(labels, datas): # show label and data of each class
                # forward propagate through the network
                ho, _, h1, _ = self.forward_propagate(sample)
                err.append(self.inst_err(label, ho)) # store instantenous error of each example

                # update processs
                self.backward_propagate(ho, h1, label, sample)
            err_epoch.append(np.average(err)) # store average error per epoch

            # find validation loss after each epoch
            valid_loss_this_epoch = []
            for label, sample in zip(valid_labels, valid_data):
                ho, _, _, _ = self.forward_propagate(sample)
                valid_loss_this_epoch.append(np.average(self.inst_err(label, ho)))
            valid_loss_epoch.append(np.average(valid_loss_this_epoch))

        return err_epoch, valid_loss_epoch
        
    def fit_classifier(self, class1, class2, class3, class1_valid, class2_valid, class3_valid):
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
        
        # validation set
        all_valid = np.concatenate((class1_valid, class2_valid, class3_valid), axis=0)
        all_true_labels_valid = np.concatenate((labels[0]* class1_valid.shape[0],
                                            labels[1]* class2_valid.shape[0],
                                            labels[2]* class3_valid.shape[0]), axis=0)
        return self.fit(all_data, all_true_labels, all_valid, all_true_labels_valid)
    
    def fit_regressor(self, data, valid):
        '''
        This model fits the model for regression
        '''
        target = data[:, -1]
        features = data[:, :-1].reshape((data.shape[0], data.shape[1]-1)) # for the sake of uniformity

        valid_target = valid[:, -1]
        valid_features = valid[:, :-1].reshape((valid.shape[0], valid.shape[1]-1)) # for the sake of uniformity

        return self.fit(features, target, valid_features, valid_target)
            
    def predict(self, sample):
        ho, _, _, _ = self.forward_propagate(sample)
        return np.argmax(ho) + 1
        
    def regress(self, sample):
        ho, _, _, _ = self.forward_propagate(sample)
        return ho

    
class fcnn_2layer:
    def __init__(self,
                 node_layers = [2, 5, 3, 3],
                 n_hidden_layers = 2,
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
        self.nodes_h1_dim = node_layers[1]
        self.nodes_h2_dim = node_layers[2]
        self.output_dim = node_layers[-1]
        # initializing weights
        self.Wh1 = np.random.randn(self.input_dim+1, self.nodes_h1_dim) # a = W.T @ x; rows of Wh1 is the dimention of input
        self.Wh2 = np.random.randn(self.nodes_h1_dim, self.nodes_h2_dim) # a = W.T @ x; rows of Wh2 is the dimention of input
        self.Wo = np.random.randn(self.nodes_h2_dim, self.output_dim) # a = W.T @ x; rows of Wh is the dimention of input
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
        ho, a0, h2, a2, h1, a1 (returns in reverse order of operation)
        '''
        # input layer
        x_hat = np.concatenate(([1], x)) # augmenting the input vector
        # hidden layer 1
        a1 = np.matmul(self.Wh1.T, x_hat) # hidden layer 1 activation value
        h1 = self.logistic_sigmoid(a1) # hidden layer 1 signal

        # hidden layer 2
        a2 = np.matmul(self.Wh2.T, h1) # hidden layer 2 activation value
        h2 = self.logistic_sigmoid(a2) # hidden layer 2 signal

        #output layer
        ao = np.matmul(self.Wo.T, h2) # output layer activation value
        if self.output_activation == "sigmoid":
            ho = self.logistic_sigmoid(ao) # output layer signal
        elif self.output_activation == "linear":
            ho = ao

        return ho, ao, h2, a2, h1, a1

    def backward_propagate(self, ho, h2, h1, label, sample):
        '''
        This is the meat of the FCNN
        
        Parameter:
        Signal vectors of each layer in reverse order, true label, sample
        
        Returns:
        None'''
        x_hat = np.concatenate(([1], sample)) # augmenting the input vector

        if self.output_activation == "sigmoid":
            delta_o = ((label-ho)*ho*(1-ho))
        elif self.output_activation == "linear":
            delta_o = (label-ho)

        # update output layer weights 
        self.Wo += self.learning_rate*np.outer(h2, delta_o)

        #update hidden layer 2 weights
        delta_h2 = np.matmul(self.Wo, delta_o)*h2*(1-h2)
        self.Wh2 += self.learning_rate*np.outer(h1, delta_h2)

        #update hidden layer 1 weights
        delta_h1 = np.matmul(self.Wh2, delta_h2)*h1*(1-h1)
        self.Wh1 += self.learning_rate*np.outer(x_hat, delta_h1)

        return None
    
    def fit(self, datas, labels, valid_data, valid_labels):
        '''
        This method fits the model to the required data
        '''
        err_epoch = []
        valid_loss_epoch = []
        # run epochs; stopping criteria 1
        for epoch in range(self.max_epoch):
            err = [] # store instantaneous error per example
            for label, sample in zip(labels, datas): # show label and data of each class
                # forward propagate through the network
                ho, _, h2, _, h1, _ = self.forward_propagate(sample)
                err.append(self.inst_err(label, ho)) # store instantenous error of each example

                # update processs
                self.backward_propagate(ho, h2, h1, label, sample)
            err_epoch.append(np.average(err)) # store average error per epoch

            # find validation loss after each epoch
            valid_loss_this_epoch = []
            for label, sample in zip(valid_labels, valid_data):
                ho, _, _, _, _, _ = self.forward_propagate(sample)
                valid_loss_this_epoch.append(np.average(self.inst_err(label, ho)))
            valid_loss_epoch.append(np.average(valid_loss_this_epoch))
            
        return err_epoch, valid_loss_epoch
        
    def fit_classifier(self, class1, class2, class3, class1_valid, class2_valid, class3_valid):
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
        
        # validation set
        all_valid = np.concatenate((class1_valid, class2_valid, class3_valid), axis=0)
        all_true_labels_valid = np.concatenate((labels[0]* class1_valid.shape[0],
                                            labels[1]* class2_valid.shape[0],
                                            labels[2]* class3_valid.shape[0]), axis=0)
        return self.fit(all_data, all_true_labels, all_valid, all_true_labels_valid)
    
    def fit_regressor(self, data, valid):
        '''
        This model fits the model for regression
        '''
        target = data[:, -1]
        features = data[:, :-1].reshape((data.shape[0], data.shape[1]-1)) # for the sake of uniformity

        valid_target = valid[:, -1]
        valid_features = valid[:, :-1].reshape((valid.shape[0], valid.shape[1]-1)) # for the sake of uniformity

        return self.fit(features, target, valid_features, valid_target)
    
    def predict(self, sample):
        ho, _, _, _, _, _ = self.forward_propagate(sample)
        return np.argmax(ho) + 1
    
    def regress(self, sample):
        ho, _, _, _, _, _ = self.forward_propagate(sample)
        return ho

