import numpy as np
from matplotlib import pyplot as plt
# import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
import copy
import pickle

 
# # Importing data


def data_import(data_path):
    class_labels = os.listdir(data_path) # reads directory names as class-labels
    data=[]
    labels=[]
    for class_ in class_labels:
        if class_ == '.DS_Store':
            continue
        class_path = data_path+'/'+class_
        imgs = os.listdir(class_path) # reads images names to read
        for img in imgs:
            if img == '.DS_Store':
                continue
            data.append(cv2.imread(class_path+'/'+img, cv2.IMREAD_GRAYSCALE))
            labels.append(int(class_))

    return np.array(data), np.array(labels)


test_path='./Group_10/test'
train_path='./Group_10/train'
val_path='./Group_10/val'
# test_data, test_labels = data_import(test_path)
# train_data, train_labels = data_import(train_path)
# val_data, val_labels = data_import(val_path)

# with open('test_data', mode='wb') as f:
#     pickle.dump(test_data, f)
# with open('train_data', mode='wb') as f:
#     pickle.dump(train_data, f)
# with open('val_data', mode='wb') as f:
#     pickle.dump(val_data, f)

# with open('test_labels', mode='wb') as f:
#     pickle.dump(test_labels, f)
# with open('train_labels', mode='wb') as f:
#     pickle.dump(train_labels, f)
# with open('val_labels', mode='wb') as f:
#     pickle.dump(val_labels, f)

with open('test_data', mode='rb') as f:
    test_data = pickle.load(f)
with open('train_data', mode='rb') as f:
    train_data = pickle.load(f)
with open('val_data', mode='rb') as f:
    val_data = pickle.load(f)

with open('test_labels', mode='rb') as f:
    test_labels = pickle.load(f)
with open('train_labels', mode='rb') as f:
    train_labels = pickle.load(f)
with open('val_labels', mode='rb') as f:
    val_labels = pickle.load(f)

print('Summary of data')
print(f'No. of train images: {len(train_data)}')
print(f'No. of test images: {len(test_data)}')
print(f'No. of val images: {len(val_data)}')

 
# ## SGD with Momentum

 
# ### Model


initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=6)
# Three layer FCNN
# 250,400,100
model_3 = keras.Sequential([
    Flatten(input_shape=(28,28), name='Input_layer'), # image data as input
    Dense(250, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_1'),
    Dense(400, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_2'),
    Dense(100, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_3'),
    Dense(10, activation='softmax', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Output')
], name='FCNN_3layer')
model_3.summary()


earlystopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=2, verbose=1)
# modelCheckpointsSGD_momentum = keras.callbacks.ModelCheckpoint(filepath='./modelCheckpoints/SGD_momentum/model.{epoch:02d}-{loss:.2f}.h5', verbose=0)
sgd_moment_optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,name='SGD_moment')
model_3.compile(optimizer=sgd_moment_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


history = model_3.fit(x=train_data, y=train_labels, batch_size=1, epochs=100_000,
                    callbacks=[earlystopping],
                    verbose=1, shuffle=True,
                    validation_split=0.0, validation_data=(val_data, val_labels), validation_batch_size=None)

model_3.save('SGD-moment-250,400,100.h5', overwrite=False, include_optimizer=True)
# plt.figure()
# plt.title("SGD with momentum")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")

# plt.plot(history.history['loss'], label='Training')
# plt.plot(history.history['val_loss'], label='Validation')
# plt.legend()

 
# # ### Test


# model_3.evaluate(test_data, test_labels)


# pred_labels = model_3.predict(test_data, verbose=0)
# pred_labels = np.argmax(pred_labels, axis=1)

# confusion_matrix = tf.math.confusion_matrix(test_labels, pred_labels, num_classes=10)
# print('(SGD with momentum)Confusion matrix on test data:\n')
# print(confusion_matrix.numpy())


# 125,250,100
model_3 = keras.Sequential([
    Flatten(input_shape=(28,28), name='Input_layer'), # image data as input
    Dense(125, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_1'),
    Dense(250, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_2'),
    Dense(100, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_3'),
    Dense(10, activation='softmax', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Output')
], name='FCNN_3layer')
model_3.summary()


earlystopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=2, verbose=1)
# modelCheckpointsSGD_momentum = keras.callbacks.ModelCheckpoint(filepath='./modelCheckpoints/SGD_momentum/model.{epoch:02d}-{loss:.2f}.h5', verbose=0)
sgd_moment_optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,name='SGD_moment')
model_3.compile(optimizer=sgd_moment_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


history = model_3.fit(x=train_data, y=train_labels, batch_size=1, epochs=100_000,
                    callbacks=[earlystopping],
                    verbose=1, shuffle=True,
                    validation_split=0.0, validation_data=(val_data, val_labels), validation_batch_size=None)
model_3.save('SGD-moment-125,250,100.h5', overwrite=False, include_optimizer=True)


# 784,500,100
model_3 = keras.Sequential([
    Flatten(input_shape=(28,28), name='Input_layer'), # image data as input
    Dense(784, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_1'),
    Dense(500, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_2'),
    Dense(100, activation='sigmoid', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Hidden_layer_3'),
    Dense(10, activation='softmax', kernel_initializer=initializer, bias_initializer=keras.initializers.Zeros(), name='Output')
], name='FCNN_3layer')
model_3.summary()


earlystopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=2, verbose=1)
# modelCheckpointsSGD_momentum = keras.callbacks.ModelCheckpoint(filepath='./modelCheckpoints/SGD_momentum/model.{epoch:02d}-{loss:.2f}.h5', verbose=0)
sgd_moment_optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,name='SGD_moment')
model_3.compile(optimizer=sgd_moment_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


history = model_3.fit(x=train_data, y=train_labels, batch_size=1, epochs=100_000,
                    callbacks=[earlystopping],
                    verbose=1, shuffle=True,
                    validation_split=0.0, validation_data=(val_data, val_labels), validation_batch_size=None)
model_3.save('SGD-moment-784,500,100.h5', overwrite=False, include_optimizer=True)
