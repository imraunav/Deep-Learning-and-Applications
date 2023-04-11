import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import random, clip_by_value
import pickle

from helper_fn import data_import, relabel


def onehiddenlayer(train_data, noisy_factor):
    n_components = 128
    inputs = Input(shape=(784,), name="Encoder-input")
    h = Dense(n_components, activation="tanh", name="Bottleneck-layer")(inputs)
    outputs = Dense(784, activation="linear", name="Decoder-output")(h)

    encoder = Model(inputs=inputs, outputs=h, name=f'Encoder{n_components}')
    decoder = Model(inputs=h, outputs=outputs, name=f'Decoder{n_components}')
    autoencoder = Model(inputs=inputs, outputs=outputs, name=f'Autoencoder{n_components}')
    adam_optimizer = Adam(learning_rate = 0.001)

    autoencoder.compile(optimizer=adam_optimizer,
                    loss="mse",
                    metrics=['accuracy'])
    earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=10,
                                verbose=1)
    autoencoder.summary()
    history = autoencoder.fit(x=train_data, y=train_data,
                            batch_size=32, epochs=100_000,
                            callbacks=[earlystopping],
                            verbose=1, shuffle=True,
                            validation_split=0.0)
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Original")
    
    plt.imshow(np.array(train_data[6000, :]).reshape((28,28)))

    plt.subplot(1,2,2)
    plt.title("Autoencoder reconstruction")
    plt.imshow(autoencoder.predict(np.array(train_data[6000, :]).reshape(1, -1)).reshape((28,28)))
    # plt.suptitle(f"{n_components} bottleneck")

    with open(f'./ProgrammingAssignment4/denoising-autoencoder_models/history_1layer_{noisy_factor}.pkl', mode='wb') as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    encoder.save(filepath=f"./ProgrammingAssignment4/denoising-autoencoder_models/encoder_1layer_{noisy_factor}.h5", overwrite=True, include_optimizer=True)
    autoencoder.save(filepath=f"./ProgrammingAssignment4/denoising-autoencoder_models/autoencoder_1layer_{noisy_factor}.h5", overwrite=True, include_optimizer=True)

# using model subclassing to build the autoencoder, following example from tensorflow introduction
# class Encoder(Model):
#     def __init__(self, latent_dim) -> None:
#         super().__init__()
#         self.inputs = Input(shape=(784,), name="Encoder-input")
#         self.h = Dense(latent_dim, activation="tanh", name="Bottleneck-layer")
#         # outputs = Dense(784, activation="linear", name="Decoder-output")
#     def call(self, x):
#         return self.h(self.inputs(x))

# class Decoder(Model):
#     def __init__(self) -> None:
#         super().__init__()
#         # self.inputs = Input(shape=(784,), name="Encoder-input")
#         # self.h = Dense(latent_dim, activation="tanh", name="Bottleneck-layer")
#         self.outputs = Dense(784, activation="linear", name="Decoder-output")
#     def call(self, x):
#         return self.outputs(x)
    
# class Autoencoder(Model):
#     def __init__(self, latent_dim) -> None:
#         super().__init__()
#         # inputs = Input(shape=(784,), name="Encoder-input")
#         # h = Dense(latent_dim, activation="tanh", name="Bottleneck-layer")
#         # outputs = Dense(784, activation="linear", name="Decoder-output")
        
#         self.encoder = Encoder(latent_dim)
#         self.decoder = Decoder()

#     def call(self, x):
#         encode = self.encoder(x)
#         decode = self.decoder(encode)
#         return decode
    
def noisify(train_data, test_data, val_data, noisy_factor):
    noisy_train = train_data + noisy_factor*random.normal(train_data.shape)
    noisy_test = test_data + noisy_factor*random.normal(test_data.shape)
    noisy_val = val_data + noisy_factor*random.normal(val_data.shape)

    noisy_train = clip_by_value(noisy_train, clip_value_min=0, clip_value_max=1)
    noisy_test = clip_by_value(noisy_test, clip_value_min=0, clip_value_max=1)
    noisy_val = clip_by_value(noisy_val, clip_value_min=0, clip_value_max=1)
    return noisy_train, noisy_test, noisy_val


def main():
    test_path='./ProgrammingAssignment4/Group_10/test'
    train_path='./ProgrammingAssignment4/Group_10/train'
    val_path='./ProgrammingAssignment4/Group_10/val'

    #import datasets
    test_data, test_labels = data_import(test_path)
    train_data, train_labels = data_import(train_path)
    val_data, val_labels = data_import(val_path)

    # train_labels = relabel(train_labels)
    # test_labels = relabel(test_labels)
    # val_labels = relabel(val_labels)

    # vectorize and normalize
    train_data = train_data.reshape(-1, 784)/255.0
    test_data = test_data.reshape(-1, 784)/255.0
    val_data = val_data.reshape(-1, 784)/255.0
    
    # noisy_factor = 0.2 # 20%
    for noisy_factor in [0.2, 0.4]:
        noisy_train, noisy_test, noisy_val = noisify(train_data, test_data, val_data, noisy_factor)
        # plt.subplot(1,2,1)
        # plt.imshow(np.reshape(noisy_train[10000,:], (28,28)), cmap="gray")
        # plt.subplot(1,2,2)
        # plt.imshow(np.reshape(train_data[10000,:], (28,28)), cmap="gray")
        # plt.show()

        # model = Autoencoder(128)
        # adam_optimizer = Adam(learning_rate = 0.001)
        # model.compile(optimizer=adam_optimizer,
        #                 loss="mse",
        #                 metrics=['accuracy'])
        # earlystopping = EarlyStopping(monitor='loss',
        #                             min_delta=1e-4,
        #                             patience=10,
        #                             verbose=1)
        # model.build((784,))
        # model.summary()
        # history = model.fit(x=noisy_train, y=train_data,
        #                         batch_size=32, epochs=100_000,
        #                         callbacks=[earlystopping],
        #                         verbose=1, shuffle=True,
        #                         validation_split=0.0)
        onehiddenlayer(noisy_train, noisy_factor)
    
        plt.show()
        
        
if __name__ == "__main__":
    main()
