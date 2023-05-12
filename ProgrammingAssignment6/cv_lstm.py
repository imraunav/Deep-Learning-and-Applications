import tensorflow as tf
from tensorflow.keras import Model, Input, layers, optimizers, callbacks, Sequential, losses
import numpy as np
import matplotlib.pyplot as plt
import pickle

from preprocessing import process_mfcc, data_import_cv

def make_lstm():
    features=39
    seq_len = 79 # fixing for the sake of sanity
    inputs = Input((seq_len, features))
    x = layers.Masking(mask_value=100_000.0)(inputs)
    x = layers.LSTM(64, return_state=False, dropout=0.2)(x)
    # x = layers.Concatenate()(x)
    outputs = layers.Dense(5, activation="Softmax")(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model



# def make_lstm2():
#     features=39
#     seq_len = 79 # fixing for the sake of sanity
#     inputs = Input((seq_len, features))
#     x = layers.Masking(mask_value=100_000.0)(inputs)
#     x = layers.LSTM(32, return_sequences=True, dropout=0.2)(x)
#     x = layers.LSTM(16, return_state=False)(x)
#     # x = layers.Concatenate()(x)
#     outputs = layers.Dense(5, activation="Softmax")(x)
#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model


def make_lstm3():
    features=39
    seq_len = 79 # fixing for the sake of sanity
    inputs = Input((seq_len, features))
    x = layers.Masking(mask_value=100_000.0)(inputs)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(32, return_state=False)(x)
    # x = layers.Concatenate()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(5, activation="Softmax")(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def main():
    n = 3 # model number
    # rnn = make_lstm()
    # rnn = make_lstm2()
    rnn = make_lstm3()
    data_path = "./ProgrammingAssignment6/CS671-DLA-Assignment4-Data-2022/CV_Data"
    x_train, y_train, x_test, y_test = data_import_cv(data_path)
    # print(len(y_train))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(x_train[i])
    # plt.show()
    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train, dtype="float64", padding="post", value=100_000.0)
    y_train = tf.constant(y_train)


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
    rnn.save(filepath=f"./ProgrammingAssignment6/models/cv_lstm{n}.h5", overwrite=True, include_optimizer=True)
    with open(f"./ProgrammingAssignment6/logs/hist_cvlstm{n}.pkl", mode="wb") as f:
        pickle.dump(log.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    return None
if __name__ == "__main__":
    main()