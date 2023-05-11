import tensorflow as tf
from tensorflow.keras import Model, Input, layers, optimizers, callbacks, Sequential, losses
import numpy as np
import matplotlib.pyplot as plt
import pickle

from preprocessing import process_mfcc, data_import_cv

# def make_rnn():
#     features=39
#     seq_len = 79 # fixing for the sake of sanity
#     model = Sequential()
#     model.add(layers.Input(shape=(seq_len, features), dtype=np.float64))
#     model.add(layers.Masking(mask_value=100_000.0)) #, input_shape=(seq_len, features)
#     model.add(layers.SimpleRNN(64, return_sequences=True))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(5, activation="Softmax"))
#     return model
# def make_rnn2():
#     features=39
#     seq_len = 79 # fixing for the sake of sanity
#     model = Sequential()
#     model.add(layers.Input((seq_len, features)))
#     model.add(layers.Masking(mask_value=100_000.0))
#     model.add(layers.SimpleRNN(32, return_sequences=True))
#     model.add(layers.SimpleRNN(16, return_sequences=True))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(5, activation="Softmax"))
#     return model
def make_rnn3():
    features=39
    seq_len = 79 # fixing for the sake of sanity
    model = Sequential()
    model.add(layers.Input((seq_len, features)))
    model.add(layers.Masking(mask_value=100_000.0))
    model.add(layers.SimpleRNN(32, return_sequences=True))
    model.add(layers.SimpleRNN(16, return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(5, activation="Softmax"))
    return model

def main():
    n = 2 # model number
    data_path = "./ProgrammingAssignment6/CS671-DLA-Assignment4-Data-2022/CV_Data"
    x_train, y_train, x_test, y_test = data_import_cv(data_path)
    # print(len(y_train))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(x_train[i])
    # plt.show()
    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train, dtype="float64", padding="post", value=100_000.0)
    y_train = tf.constant(y_train)

    # rnn = make_rnn()
    # rnn = make_rnn2()
    rnn = make_rnn3()

    rnn.summary()
    adam_optimizer = optimizers.Adam()
    categorical_loss = losses.SparseCategoricalCrossentropy()
    rnn.compile(optimizer=adam_optimizer, loss=categorical_loss, metrics=["accuracy"])
    earlystopping = callbacks.EarlyStopping(monitor='loss',
                                            min_delta=1e-4,
                                            patience=20,
                                            verbose=1)
    # remotemonitor=tf.keras.callbacks.RemoteMonitor(root="http://localhost:9000")
    # terminate_on_nan = callbacks.TerminateOnNaN()
    # stopping_criteria = StoppingCriteria(patience=1, min_delta=1e-4)
    log = rnn.fit(x=x_train_padded, y=y_train,  
                verbose=2, shuffle=True,
                callbacks=[earlystopping],
                epochs=10_000, validation_split=0)
    rnn.save(filepath=f"./ProgrammingAssignment6/models/cv_rnn{n}.h5", overwrite=True, include_optimizer=True)
    with open(f"./ProgrammingAssignment6/logs/hist_cvrnn{n}.pkl", mode="wb") as f:
        pickle.dump(log.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    return None
if __name__ == "__main__":
    main()