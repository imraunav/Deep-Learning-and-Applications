import numpy as np

''' Thoughts while writing this:
- Maybe trying to generalize the overall structure is making this more difficult. I should just try to do one task first.
- Starting with simple threshold logic for the activation fn and avoiding all cleverness
- Linearly Seperable data works with this perceptron 
'''
class perceptron:
    def __init__(self, labels, n_features, max_epoch=1000, activation='threshold', tol= 1e-3):
        np.random.seed(6)
        self.max_epoch = max_epoch
        self.labels1 = labels[0]
        self.labels2 = labels[1]
        self.w = np.random.randn(n_features)
        self.bias = np.random.randn(1)
        self.tol = tol
        self.activation = activation
        return None

    def activation_fn(self, activation_value):
        if self.activation == 'threshold':
            # threshold logic
            if activation_value > 0:
                return 1
            else:
                return -1
        elif self.activation == 'sigmoid':
            # sigmoid(tan hyperbolic logic)
            return np.tanh(activation_value)
        elif self.activation == 'linear':
            # linear logic
            return activation_value
        else:
            # unknown activation function
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
            learning_rate = 1/(epoch+1) #constant learning rate
            err_count = 0
            # see all class 1 data
            for sample in class1_data:
                pred_label = self.predict(sample)
                if pred_label != self.labels1:
                    self.w += learning_rate*sample
                    self.bias += learning_rate
                    err_count += 1
            # see all class 2 data
            for sample in class2_data:
                pred_label = self.predict(sample)
                if pred_label != self.labels2:
                    self.w -= learning_rate*sample
                    self.bias -= learning_rate
                    err_count += 1
            error = err_count/(class1_data.shape[0] + class2_data.shape[0])
            err_epoch.append(error)
            # check convergence
            if error <= self.tol:
                break

        return err_epoch

    def predict(self, input):
        # augmenting the input vector
        activation_value = np.dot(self.w, input) + self.bias
        signal = self.activation_fn(activation_value)
        if signal > 0:
            return self.labels1
        elif signal <= 0:
            return self.labels2 



# class perceptron:
#     def __init__(self, n_feature=2, activation='threshold', max_epoch = 1000):
#         self.w = np.ones((n_feature+1,)) # augmented weight vector initialised with thereshold
#         self.labels = (-1, 1) # will use tanh for sigmoid function as activation function
#         self.activation = activation
#         self.max_epoch = max_epoch
#         return None

#     def fit(self, train_data, train_label):
#         '''
#         This method will take the training data and return the mean squared error per epoch after training the weights of the perceptron
        
#         Parameters:
#         train_data - all training data to train the perceptron on
        
#         Return:
#         error per epoch
#         '''
#         self.w = np.random.uniform(low=train_data.min(), high=train_data.max(), size=(3,))
#         err_epoch = []
#         for epoch in range(self.max_iter):
#             err_count = 0
#             learning_rate = 1/epoch # an approch suggested in class to non-leanearize the learning-rate
            
#             for sample, true_label in zip(train_data, train_label):
#                 pred_label = self.predict(sample) # predict for each sample
#                 self.w += learning_rate*(true_label-pred_label)*sample # correct for each sample
#                 if true_label != pred_label:
#                     err_count += 1
            
#             err_epoch.append(err_count)
            
#             #convergence criterion
#             if err_count == 0:
#                 break
#         return err_epoch
        

#     def predict(self, input):
#         '''
#         This method will return the predictin based on the trained weights for the given input vector
        
#         Parameters:
#         input - input test datapoint for prediction

#         Return:
#         class/value predicted
#         '''
#         #augmenting the input vector to accomodate the bias
#         x = np.concatenate([[1], [input]], axis=0)
#         activation_value = np.dot(self.w, x)
#         y_hat = self.activation_fn(activation_value) #predict a label based on activation value
#         return y_hat

#     def activation_fn(self, a):
#         if self.activation == 'threshold':
#             if a > 0:
#                 return 1
#             else:
#                 return -1
#         # elif self.activation == 'sigmoid':
#         #     return np.tanh(a) 
#         # elif self.activation == 'linear':
#         #     return a
#         else:
#             raise TypeError('Invalid activation function')

