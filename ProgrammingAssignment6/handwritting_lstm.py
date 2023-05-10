import tensorflow as tf
from tensorflow.keras import Model, Input, layers, optimizers, callbacks, Sequential, losses
import numpy as np
import matplotlib.pyplot as plt
import pickle

from preprocessing import process_handwriting, data_import_handwriting, StoppingCriteria

def make_lstm():
    features=2
    seq_len = 200 # fixing for the sake of sanity
    model = Sequential()
    model.add(layers.Masking(mask_value=-1, input_shape=(seq_len, features)))
    model.add(layers.LSTM(5, input_shape=(seq_len, features)))
    model.add(layers.Dense(5, activation="Softmax"))
    return model
# def make_rnn2():
#     features=2
#     seq_len = 200 # fixing for the sake of sanity
#     model = Sequential()
#     model.add(layers.Masking(mask_value=-1, input_shape=(None, features)))
#     model.add(layers.SimpleRNN(10, return_sequences=True))
#     model.add(layers.SimpleRNN(5))
#     model.add(layers.Dense(5, activation="Softmax"))
#     return model
# def make_rnn3():
#     features=2
#     seq_len = 200 # fixing for the sake of sanity
#     model = Sequential()
#     model.add(layers.Masking(mask_value=-1, input_shape=(None, features)))
#     model.add(layers.SimpleRNN(10, input_shape=(None, features)))
#     # model.add(layers.SimpleRNN(15))
#     # model.add(layers.Dense(10, activation='relu'))
#     # model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(5, activation="Softmax"))
#     return model
def main():
    tf.random.set_seed(32)
    n = 1 # model number
    data_path = "./ProgrammingAssignment6/CS671-DLA-Assignment4-Data-2022/Handwriting_Data"
    x_train, y_train, x_test, y_test = data_import_handwriting(data_path)
    # y_train = tf.keras.utils.to_categorical(y_train)
    # y_test = tf.keras.utils.to_categorical(y_test)
    y_train = tf.constant(y_train)
    y_test = tf.constant(y_test)
    
    # print(x_train[120].shape)
    # print(np.min(x_train[100]), np.max(x_train[100]))
    mask_val = -1   # number unlikely to appear in normalized data
    x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
    x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
    lstm = make_lstm()
    # rnn = make_rnn2()
    # rnn = make_rnn3()

    lstm.summary()
    adam_optimizer = optimizers.Adam(learning_rate=0.1)
    categorical_loss = losses.SparseCategoricalCrossentropy()
    lstm.compile(optimizer=adam_optimizer, loss=categorical_loss, metrics=["accuracy"])
    earlystopping = callbacks.EarlyStopping(monitor='loss',
                                            min_delta=0.0001,
                                            patience=10,
                                            verbose=1)
    # remotemonitor=tf.keras.callbacks.RemoteMonitor(root="http://localhost:9000")
    # terminate_on_nan = callbacks.TerminateOnNaN()
    # stopping_criteria = StoppingCriteria(patience=1, min_delta=1e-4)
    log = lstm.fit(x=x_train_padded, y=y_train,  
                verbose=2, shuffle=True,
                callbacks=[earlystopping],
                epochs=10_000, validation_split=0)
    lstm.save(filepath=f"./ProgrammingAssignment6/models/handwriting_lstm{n}.h5", overwrite=True, include_optimizer=True)
    with open(f"./ProgrammingAssignment6/logs/hist_handlstm{n}.pkl", mode="wb") as f:
        pickle.dump(log.history, f, protocol=pickle.HIGHEST_PROTOCOL)



    return None
if __name__ == "__main__":
    main()