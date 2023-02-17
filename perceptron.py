import numpy as np

''' Thoughts while writing this:
- Maybe trying to generalize the overall structure is making this more difficult. I should just try to do one task first.
- Starting with simple threshold logic for the activation fn and avoiding all cleverness
- Linearly Seperable data works with this perceptron 
- Having trouble with sigmoid activation function for learning the weights. The application of gradient descent algorithm with respect to the true labels may be confusing me.
'''
class perceptron:
    def __init__(self, labels, n_features, max_epoch=1000, activation='sigmoid', tol= 1e-3, learning_rate=0.25):
        # np.random.seed(100)
        self.max_epoch = max_epoch
        self.labels1 = labels[0]
        self.labels2 = labels[1]
        self.w = np.random.randn(n_features)
        self.bias = np.random.randn(1)
        # self.w = np.array([1]*n_features, dtype=np.float64)
        # self.bias = 1.0
        self.tol = tol
        self.activation = activation
        self.learning_rate = learning_rate
        return None
    
    def logictic_sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def activation_fn(self, activation_value):
        if self.activation == 'sigmoid':
            # sigmoid(tan hyperbolic logic)
            return np.tanh(activation_value)
            # return self.logictic_sigmoid(activation_value)
        elif self.activation == 'linear':
            # linear logic
            return activation_value
        
    def grad_descent(self, true_label, signal, sample):
        if self.activation == 'sigmoid':
            delta = (true_label-signal)*(1-(signal**2))
        elif self.activation == 'linear':
            delta = (true_label-signal)
        self.w += self.learning_rate*delta*sample
        self.bias += self.learning_rate*delta
        return None

    def fit(self, class1_data, class2_data):
        '''
        This method will take the training data and return the mean squared error per epoch after training the weights of the perceptron
        
        Parameters:
        train_data - all training data to train the perceptron on
        
        Return:
        error per epoch
        '''
        err_epoch = []
        for epoch in range(self.max_epoch):
            learning_rate = self.learning_rate #constant learning rate
            err_collect = []
            # see all class 1 data(negetive class)
            true_label = -1
            for sample in class1_data:
                signal = self.predict_signal(sample)
                err_collect.append(0.5*np.square(true_label-signal))
                self.grad_descent(true_label, signal, sample)
            # see all class 2 data(positive class)
            true_label = 1
            for sample in class2_data:
                signal = self.predict_signal(sample)
                err_collect.append(0.5*np.square(true_label-signal))
                self.grad_descent(true_label, signal, sample)
            avg_error = np.mean(err_collect)
            err_epoch.append(avg_error)
            # check convergence
            # if avg_error <= self.tol:
            #     break

        return err_epoch

    def predict_signal(self, input):
        '''
        This function generates the signal from the perceptron

        Parameter:
        Input - Input sample for the perceptron

        Return:
        Signal - Signal generated from perceptron after activation function
        '''
        activation_value = np.dot(self.w, input) + self.bias
        signal = self.activation_fn(activation_value)
        return signal 

    def predict(self, input):
        '''
        Parameter:
        Input - Input sample for the perceptron

        Return:
        Class label of the predicted class
        '''
        signal = self.predict_signal(input)
        if signal < 0:
            return self.labels1
        elif signal >= 0:
            return self.labels2 
    def regress(self, input):
        return self.predict_signal(input)



