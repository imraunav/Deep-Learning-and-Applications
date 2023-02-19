import numpy as np
'''
Observations
- Compared to previous attempt, this perceptron works best due to two major changes:
    * The initialisation is not random anymore. Instead this initialises with unit slope equivalent and unit intercept.
    * This perceptron is more general and tweaking in the code need to be only in the main code, rather than this perticular class definition.
    * Special care should be given to the 'learning_rate' value, especially for learning a non-linearly seperable data. There are chances of skipping narrow-and-deep valleys for larger learning rates, which will cause the error per epoch to even rise instead of falling to a suitable value. 
'''
class perceptron:
    def __init__(self, n_features, labels=(1, 2), tol=1e-3, activation='sigmoid', max_epoch=1000, learning_rate=0.25):
        self.n_features = n_features
        self.label1 = labels[0] # positive class
        self.label2 = labels[1] # negetive class
        self.tol = tol
        self.activation = activation
        self.leaning_rate = learning_rate
        self.max_epoch = max_epoch
        self.w = np.full(n_features+1, fill_value=1, dtype=float) # augmented weight vector
        return None

    def get_signal(self, input):
        '''
        Gets the signal generated from the perceptron
        
        Parameters:
        Input sample

        Return:
        Signal from the perceptron
        '''
        input = np.concatenate(([1], input)) # augmenting the input vector
        activation_value = np.dot(self.w, input)
        if self.activation == 'threshold':
            if activation_value > 0.5:
                return 1
            else:
                return 0
        elif self.activation == 'sigmoid_logistic' or self.activation == 'sigmoid':
            return self.logistic_sigmoid(activation_value)
        elif self.activation == 'sigmoid_tanh':
            return np.tanh(activation_value)
        elif self.activation == 'linear':
            return activation_value

    def logistic_sigmoid(self, x):
        '''
        Logistic sigmoid function
        '''
        return 1/(1+np.exp(-x))
    
    def predict_class(self, sample):
        '''
        Return the class label based on the signal from the perceptron
        '''
        signal = self.get_signal(sample)
        if self.activation == 'sigmoid_logistic' or self.activation == 'sigmoid' or self.activation == 'threshold':
            threshold = 0.5
        elif self.activation == 'sigmoid_tanh':
            threshold = 0

        if signal < threshold:
            return self.label1
        elif signal >= threshold:
            return self.label2
        
    def fit_classification(self, class1_data, class2_data):
        '''
        This method will take the training data and return the mean squared error per epoch after training the weights of the perceptron
        
        Parameters:
        Class 1 training data, Class 2 training data
        
        Return:
        Error per epoch during training
        '''
        if self.activation == 'sigmoid_logistic' or self.activation == 'sigmoid' or self.activation == 'threshold':
            labels = (0, 1)
        elif self.activation == 'sigmoid_tanh':
            labels = (-1, 1)
        
        all_data = np.concatenate((class1_data, class2_data), axis=0)
        all_true_labels = np.concatenate((np.full(shape=class1_data.shape[0], fill_value=labels[0]),
                        np.full(shape=class2_data.shape[0], fill_value=labels[1])), axis=0)
        return self.fit(all_data, all_true_labels)

    def grad_descent(self, true_label, signal, sample):
        '''
        Update the weight vector using gradient descent method
        '''
        if self.activation == 'threshold' or self.activation == 'linear':
            delta = (true_label-signal)
        if self.activation == 'sigmoid' or self.activation == 'sigmoid_logistic':
            delta = (true_label-signal)*signal*(1-signal)
        if self.activation == 'sigmoid_tanh':
            delta = (true_label-signal)*(1-np.square(signal))
        sample = np.concatenate(([1], sample))
        self.w += self.leaning_rate*delta*sample
        return None

    def fit(self, input, output):
        '''
        This method takes in input samples and their supposed output and learn the weight to fit the output.
        
        Parameters:
        Input vectors and output labels

        Return
        error per epoch
        '''
        epoch_error = []
        for epoch in range(self.max_epoch):
            err_collect = []
            for true_label, sample in zip(output, input):
                signal = self.get_signal(sample)
                err_collect.append(0.5*np.square(true_label-signal)) # collect the instantaneous error
                self.grad_descent(true_label, signal, sample) # update weights
            avg_err = np.average(err_collect)
            epoch_error.append(avg_err) # collect average error per epoch
        return epoch_error

    def fit_regression(self, data):
        '''
        This methods is to fit the model for regression.
        
        Parameters:
        Data to regress

        Return
        Error per epoch during training
        '''
        input, output = data[:, :-1], data[:, -1:]
        input = input.reshape((data.shape[0], self.n_features)) # just for the sake of uniformality
        return self.fit(input, output)

    def predict_regress(self, input):
        '''
        Parameters
        Input vector
        
        Returns
        Output measure
        '''
        return self.get_signal(input)