import numpy as np
from matplotlib import pyplot as plt
import os
from tensorflow.keras import callbacks

def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def process_handwriting(filename):
    data = np.loadtxt(filename, dtype=np.float64)
    # print(data)
    seq_len = data[0] # number of coordinates in the file
    data = data[1:] # remove the number, contain only the coordinates
    # print(data.reshape((-1, 2))
    data = data.reshape((-1, 2))
    data[:, 0] = normalize(data[:, 0])
    data[:, 1] = normalize(data[:, 1])

    return data


def process_mfcc(filename):
    data = np.loadtxt(filename, dtype=np.float64)
    # print(data.shape)
    # data = data.reshape((-1,))
    return data


def data_import_handwriting(data_path, class_labels = None):
    if class_labels == None:
        class_labels = {"a": 0,
                        "bA": 1,
                        "chA": 2,
                        "lA": 3,
                        "tA": 4,
                        }
    else:
        class_labels = {key:i for key, i in enumerate(class_labels)}
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # print(class_labels.keys())
    for class_ in class_labels.keys():
        class_path = data_path+'/'+class_

        data_category = "train"
        files = os.listdir(class_path+'/'+data_category)
        for file in files:
            if file == '.DS_Store':
                continue
            # print(file)
            x_train.append(process_handwriting(class_path+'/'+data_category+'/'+file))
            y_train.append(class_labels[class_])

        data_category = "dev"
        files = os.listdir(class_path+'/'+data_category)
        for file in files:
            if file == '.DS_Store':
                continue
            x_test.append(process_handwriting(class_path+'/'+data_category+'/'+file))
            y_test.append(class_labels[class_])
    return x_train, y_train, x_test, y_test


def data_import_cv(data_path, class_labels=None):
    if class_labels == None:
        class_labels = {"tA" : 0,
                        "kaa" : 1,
                        "ka" : 2,
                        "sa" : 3,
                        "rI" : 4,
                        }
    else: 
        class_labels = {key:i for key, i in enumerate(class_labels)}
    
    # class_labels = os.listdir(data_path) # reads directory names as class-labels

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # print(class_labels.keys())
    for class_ in class_labels.keys():
        class_path = data_path+'/'+class_

        data_category = "train"
        files = os.listdir(class_path+'/'+data_category)
        for file in files:
            if file == '.DS_Store':
                continue
            x_train.append(process_mfcc(class_path+'/'+data_category+'/'+file))
            y_train.append(class_labels[class_])

        data_category = "test"
        files = os.listdir(class_path+'/'+data_category)
        for file in files:
            if file == '.DS_Store':
                continue
            x_test.append(process_mfcc(class_path+'/'+data_category+'/'+file))
            y_test.append(class_labels[class_])
    return x_train, y_train, x_test, y_test

class StoppingCriteria(callbacks.Callback):
    """
    Stop training when change in loss is less than 1e-4.
    """
    def __init__(self, patience=0, min_delta=1e-4):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.min_delta = min_delta
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.prevloss = np.inf # initiate with inf loss
        self.stopped_epoch = 0
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        # if np.less(current, self.best):
        #     self.best = current
        #     self.wait = 0
        #     self.best_weights = self.model.get_weights()
        # else:
        if abs(current-self.prevloss) <= self.min_delta:
            self.wait += 1 
        else:
            self.wait = 0 #restore waiting period

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_trianing = True
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: Stopped training")