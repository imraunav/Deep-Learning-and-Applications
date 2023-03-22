from tensorflow import keras
import pickle
from tensorflow.keras.layers import Flatten, Dense, Input, Rescaling
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

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


# Three layer FCNN
# 400, 200, 100
# n_node=[400, 200, 100]

# initializer1 = keras.initializers.RandomNormal(stddev=0.01, seed=3)
# initializer2 = keras.initializers.RandomNormal(stddev=0.01, seed=10)
# initializer3 = keras.initializers.RandomNormal(stddev=0.01, seed=64)
# initializer4 = keras.initializers.RandomNormal(stddev=0.01, seed=128)


# model_1 = keras.Sequential([
#     Input((28,28), name='Input_layer'), # image data as input
#     Flatten(name='Vectorize'),
#     Dense(n_node[0], activation='sigmoid', name='Hidden_layer_1', kernel_initializer=initializer1, bias_initializer='zeros'),
#     Dense(n_node[1], activation='sigmoid', name='Hidden_layer_2', kernel_initializer=initializer2, bias_initializer='zeros'),
#     Dense(n_node[2], activation='sigmoid', name='Hidden_layer_3', kernel_initializer=initializer3, bias_initializer='zeros'),
#     Dense(10, activation='softmax', kernel_initializer=initializer4, bias_initializer='zeros', name='Output'),
# ], name='FCNN_3layer')
# model_1.summary()

# # optimizer
# adam = keras.optimizers.Adam(learning_rate=0.001,
#         beta_1=0.9, beta_2=0.999, epsilon=1e-8,
#         name="Adam")
# earlystopping = keras.callbacks.EarlyStopping(monitor='loss',
#                                               min_delta=1e-4,
#                                               patience=1,
#                                               verbose=1)

# model_1.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# for layer in model_1.layers:
#     print(layer.weights)

# history = model_1.fit(x=train_data, y=train_labels,
#                     batch_size=1, epochs=100_000,
#                     callbacks=[earlystopping],
#                     verbose=1, shuffle=True,
#                     validation_data=[val_data, val_labels])

# # saving
# model_1.save(filepath="Adam_model1.h5", overwrite=True, include_optimizer=True)

# with open('Adam_history1.pkl', mode='wb') as f:
#     pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
# # plt.plot(history.history['loss'])
# # plt.xlabel('Epochs')
# # plt.ylabel('Adam') 
# # plt.show()


# # Three layer FCNN
# # 600, 300, 200
# n_node=[600, 300, 200]

# model_2 = keras.Sequential([
#     Input((28,28), name='Input_layer'), # image data as input
#     Flatten(name='Vectorize'),
#     Dense(n_node[0], activation='sigmoid', name='Hidden_layer_1', kernel_initializer=initializer1, bias_initializer='zeros'),
#     Dense(n_node[1], activation='sigmoid', name='Hidden_layer_2', kernel_initializer=initializer2, bias_initializer='zeros'),
#     Dense(n_node[2], activation='sigmoid', name='Hidden_layer_3', kernel_initializer=initializer3, bias_initializer='zeros'),
#     Dense(10, activation='softmax', kernel_initializer=initializer4, bias_initializer='zeros', name='Output'),
# ], name='FCNN_3layer')
# model_2.summary()

# # optimizer
# adam = keras.optimizers.Adam(learning_rate=0.001,
#         beta_1=0.9, beta_2=0.999, epsilon=1e-8,
#         name="Adam")
# earlystopping = keras.callbacks.EarlyStopping(monitor='loss',
#                                               min_delta=1e-4,
#                                               patience=1,
#                                               verbose=1)

# model_2.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # for layer in model_1.layers:
# #     print(layer.weights)

# history = model_2.fit(x=train_data, y=train_labels,
#                     batch_size=1, epochs=100_000,
#                     callbacks=[earlystopping],
#                     verbose=1, shuffle=True,
#                     validation_data=[val_data, val_labels])

# # saving
# model_2.save(filepath="Adam_model2.h5", overwrite=True, include_optimizer=True)

# with open('Adam_history2.pkl', mode='wb') as f:
#     pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
# # plt.plot(history.history['loss'])
# # plt.xlabel('Epochs')
# # plt.ylabel('Adam') 
# # plt.show()


# # Three layer FCNN
# # 1000, 500, 50
# n_node=[1000, 500, 50]

# model_3 = keras.Sequential([
#     Input((28,28), name='Input_layer'), # image data as input
#     Flatten(name='Vectorize'),]
#     Dense(n_node[0], activation='sigmoid', name='Hidden_layer_1', kernel_initializer=initializer1, bias_initializer='zeros'),
#     Dense(n_node[1], activation='sigmoid', name='Hidden_layer_2', kernel_initializer=initializer2, bias_initializer='zeros'),
#     Dense(n_node[2], activation='sigmoid', name='Hidden_layer_3', kernel_initializer=initializer3, bias_initializer='zeros'),
#     Dense(10, activation='softmax', kernel_initializer=initializer4, bias_initializer='zeros', name='Output'),
# ], name='FCNN_3layer')
# model_3.summary()

# # optimizer
# adam = keras.optimizers.Adam(learning_rate=0.001,
#         beta_1=0.9, beta_2=0.999, epsilon=1e-8,
#         name="Adam")
# earlystopping = keras.callbacks.EarlyStopping(monitor='loss',
#                                               min_delta=1e-4,
#                                               patience=1,
#                                               verbose=1)

# model_3.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # for layer in model_1.layers:
# #     print(layer.weights)

# history = model_3.fit(x=train_data, y=train_labels,
#                     batch_size=1, epochs=100_000,
#                     callbacks=[earlystopping],
#                     verbose=1, shuffle=True,
#                     validation_data=[val_data, val_labels])

# # saving
# model_3.save(filepath="Adam_model3.h5", overwrite=True, include_optimizer=True)

# with open('Adam_history3.pkl', mode='wb') as f:
#     pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
# plt.plot(history.history['loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Adam') 
# plt.show()

#Four layer 
n_node=[256, 128, 64, 32]
model = keras.Sequential(
    [   
        Input((28, 28), name="Input-layer"),
        Rescaling(1/255.0, name="Rescaler"),
        Flatten(name="Vectorize"),
        Dense(n_node[0], activation='sigmoid', name='Hidden-layer-1'),
        Dense(n_node[1], activation='sigmoid', name='Hidden-layer-2'),
        Dense(n_node[2], activation='sigmoid', name='Hidden-layer-3'),
        Dense(n_node[3], activation='sigmoid', name='Hidden-layer-4'),
        Dense(10, activation='softmax', name='Output')
    ]
)

model.summary()

# optimizer
adam = keras.optimizers.Adam(learning_rate=0.001,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        name="Adam")
earlystopping = keras.callbacks.EarlyStopping(monitor='loss',
                                              min_delta=1e-4,
                                              patience=1,
                                              verbose=1)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_data, y=train_labels,
                    batch_size=1, epochs=100_000,
                    callbacks=[earlystopping],
                    verbose=1, shuffle=True,
                    validation_data=[val_data, val_labels])

# saving
model.save(filepath="Adam_model4.h5", overwrite=True, include_optimizer=True)

with open('Adam_history4.pkl', mode='wb') as f:
    pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
# plt.plot(history.history['loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Adam') 
# plt.show()