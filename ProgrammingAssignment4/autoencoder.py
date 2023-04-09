import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from helper_fn import data_import, relabel

def onehiddenlayer(n_components, train_data, test_data):
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
                                patience=1,
                                verbose=1)
    autoencoder.summary()
    history = autoencoder.fit(x=train_data, y=train_data,
                            batch_size=32, epochs=100_000,
                            callbacks=[earlystopping],
                            verbose=1, shuffle=True,
                            validation_split=0.0)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title("Original")
    
    # plt.imshow(test_data[874, :].reshape((28,28)))

    # plt.subplot(1,2,2)
    # plt.title("Autoencoder reconstruction")
    # plt.imshow(autoencoder.predict(test_data[874, :].reshape(1, -1)).reshape((28,28)))
    # plt.suptitle(f"{n_components} bottleneck")
    with open(f'./ProgrammingAssignment4/autoencoder_models/history_1layer_{n_components}.pkl', mode='wb') as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    encoder.save(filepath=f"./ProgrammingAssignment4/autoencoder_models/encoder_1layer_{n_components}.h5", overwrite=True, include_optimizer=True)
    autoencoder.save(filepath=f"./ProgrammingAssignment4/autoencoder_models/autoencoder_1layer_{n_components}.h5", overwrite=True, include_optimizer=True)

    return None

def threehiddenlayer(n_components, train_data, test_data):
    inputs = Input(shape=(784,), name="Encoder-input")
    h1 = Dense(400, activation="tanh", name="Encoder-hidden")(inputs)
    h2 = Dense(n_components, activation="tanh", name="Bottleneck-layer")(h1)
    h3 = Dense(400, activation="tanh", name="Decoder-hidden")(h2)
    outputs = Dense(784, activation="linear", name="Decoder-output")(h3)

    encoder = Model(inputs=inputs, outputs=h2, name=f'Encoder{n_components}')
    decoder = Model(inputs=h2, outputs=outputs, name=f'Decoder{n_components}')
    autoencoder = Model(inputs=inputs, outputs=outputs, name=f'Autoencoder{n_components}')
    adam_optimizer = Adam(learning_rate = 0.001)

    autoencoder.compile(optimizer=adam_optimizer,
                    loss="mse",
                    metrics=['accuracy'])
    earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
    autoencoder.summary()
    history = autoencoder.fit(x=train_data, y=train_data,
                            batch_size=32, epochs=100_000,
                            callbacks=[earlystopping],
                            verbose=1, shuffle=True,
                            validation_split=0.0)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title("Original")
    
    # plt.imshow(test_data[874, :].reshape((28,28)))

    # plt.subplot(1,2,2)
    # plt.title("Autoencoder reconstruction")
    # plt.imshow(autoencoder.predict(test_data[874, :].reshape(1, -1)).reshape((28,28)))
    # plt.suptitle(f"{n_components} bottleneck")
    with open(f'./ProgrammingAssignment4/autoencoder_models/history_3layer_{n_components}.pkl', mode='wb') as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    encoder.save(filepath=f"./ProgrammingAssignment4/autoencoder_models/encoder_3layer_{n_components}.h5", overwrite=True, include_optimizer=True)
    autoencoder.save(filepath=f"./ProgrammingAssignment4/autoencoder_models/autoencoder_3layer_{n_components}.h5", overwrite=True, include_optimizer=True)

    return None

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

    iter_components = [32, 64, 128, 256]
    for n_components in iter_components:
        # One hidden layer autoencoder
        onehiddenlayer(n_components, train_data, test_data)

        #three hidden layer autoencoder
        threehiddenlayer(n_components, train_data, test_data)
        
        # plt.show()

    return None
if __name__ == "__main__":
    main()