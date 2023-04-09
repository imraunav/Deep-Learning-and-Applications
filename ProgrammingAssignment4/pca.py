import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
# import tensorflow_transform as tft
# import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from helper_fn import data_import, relabel
from sklearn.decomposition import PCA


"""
NOTES:
- Not using my version of PCA. Instead using the implementation by Sklearn library.
"""

def main():    
    test_path='./ProgrammingAssignment4/Group_10/test'
    train_path='./ProgrammingAssignment4/Group_10/train'
    val_path='./ProgrammingAssignment4/Group_10/val'

    #import datasets
    test_data, test_labels = data_import(test_path)
    train_data, train_labels = data_import(train_path)
    val_data, val_labels = data_import(val_path)

    train_labels = relabel(train_labels)
    test_labels = relabel(test_labels)
    val_labels = relabel(val_labels)

    # print(f"Train data shape: {train_data.shape}")
    # print(f"Test data shape: {test_data.shape}")
    # print(f"Val data shape: {val_data.shape}")

    n_train, _, _ = train_data.shape # number of training examples
    n_test, _, _ = test_data.shape # number of test examples
    n_val, _, _ = val_data.shape # number of val examples

    # Vectorization and normalization of values
    train_data = train_data.reshape(n_train, -1)/255.0
    test_data = test_data.reshape(n_test, -1)/255.0
    val_data = val_data.reshape(n_val, -1)/255.0

    # print(train_data.shape)

    mean_vec = np.mean(train_data, axis=0) # 784 dimentional vector
    # print(mean_vec.shape) 

    # mean correction
    train_data = train_data - mean_vec
    test_data = test_data - mean_vec
    val_data = val_data - mean_vec

    # perform task for all the specified components
    # pca = my_pca(train_data)

    iter_components = [32, 64, 128, 256]
    for n_components in iter_components:
        
        pca = PCA(n_components=n_components) # define pca
        pca.fit(train_data) # perform pca on training data

        reduced_train = pca.transform(train_data)
        reduced_test = pca.transform(test_data)
        reduced_val = pca.transform(val_data)

        # fcnn
        inputs = Input(shape=(n_components,), name="Input")
        x = Dense(512, activation='tanh', name="Layer1")(inputs)
        x = Dense(256, activation='tanh', name="Layer2")(x)
        x = Dense(128, activation='tanh', name="Layer3")(x)
        # x = Dense(64, activation='tanh', name="Layer4")(x)
        outputs = Dense(5, activation='softmax', name="Output")(x)
        pcamodel = Model(inputs=inputs, outputs=outputs, name=f"Model-PCA{n_components}")
        pcamodel.summary()

        adam_optimizer = Adam(learning_rate = 0.001)

        pcamodel.compile(optimizer=adam_optimizer,
                         loss="sparse_categorical_crossentropy",
                         metrics=['accuracy'])
        earlystopping = EarlyStopping(monitor='loss',
                                              min_delta=1e-4,
                                              patience=1,
                                              verbose=1)
        history = pcamodel.fit(x=reduced_train, y=train_labels,
                               batch_size=32, epochs=100_000,
                               callbacks=[earlystopping],
                               verbose=1, shuffle=True,
                               validation_split=0.0)
        with open(f'./ProgrammingAssignment4/pca_models/history_pca{n_components}.pkl', mode='wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
        pcamodel.save(filepath=f"./ProgrammingAssignment4/pca_models/pcamodel_{n_components}.h5", overwrite=True, include_optimizer=True)
        print()
        _, acc_train = pcamodel.evaluate(reduced_train, train_labels)
        _, acc_test = pcamodel.evaluate(reduced_test, test_labels)
        _, acc_val = pcamodel.evaluate(reduced_val, test_labels)

        print("Model evaluation")
        print("="*60)
        print(f"Training accuracy: {acc_train}")
        print(f"Testing accuracy: {acc_test}")
        print(f"Validation accuracy: {acc_val}")
        print("*"*60)

        


if __name__ == "__main__":
    main()