import tensorflow as tf
from tensorflow.keras import Model, Input, layers, optimizers, callbacks, Sequential, losses
import numpy as np
import matplotlib.pyplot as plt
import pickle

from preprocessing import process_handwriting, data_import_handwriting, StoppingCriteria

# def make_rnn():
#     features=2
#     seq_len = 200 # fixing for the sake of sanity
#     inputs = Input((seq_len, features))
#     x = layers.Masking(mask_value=-1)(inputs)
#     x = layers.SimpleRNN(32, return_state=False)(x)
#     # x = layers.Concatenate()(x)
#     outputs = layers.Dense(5, activation="Softmax")(x)
#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model

# def make_rnn2():
#     features=2
#     seq_len = 200 # fixing for the sake of sanity
#     inputs = Input((seq_len, features))
#     x = layers.Masking(mask_value=-1)(inputs)
#     x = layers.SimpleRNN(10, return_sequences=True)(x)
#     x = layers.SimpleRNN(5, return_state=False)(x)
#     # x = layers.Concatenate()(x)
#     outputs = layers.Dense(5, activation="Softmax")(x)
#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model

def make_rnn3():
    features=2
    seq_len = 200 # fixing for the sake of sanity
    inputs = Input((seq_len, features))
    x = layers.Masking(mask_value=-1)(inputs)
    x = layers.SimpleRNN(64, return_sequences=True)(x)
    x = layers.SimpleRNN(32, return_state=False)(x)
    # x = layers.Concatenate()(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(5, activation="Softmax")(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def main():
    tf.random.set_seed(32)
    n = 3 # model number
    # rnn = make_rnn()
    # rnn = make_rnn2()
    rnn = make_rnn3()

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

    rnn.summary()
    adam_optimizer = optimizers.Adam(learning_rate=0.0001) # Reducing the learning rate to get less oscillation 
    categorical_loss = losses.SparseCategoricalCrossentropy()
    rnn.compile(optimizer=adam_optimizer, loss=categorical_loss, metrics=["accuracy"])
    earlystopping = callbacks.EarlyStopping(monitor='loss',
                                            min_delta=0.0001,
                                            patience=20,
                                            verbose=1)
    # remotemonitor=tf.keras.callbacks.RemoteMonitor(root="http://localhost:9000")
    # terminate_on_nan = callbacks.TerminateOnNaN()
    # stopping_criteria = StoppingCriteria(patience=1, min_delta=1e-4)
    log = rnn.fit(x=x_train_padded, y=y_train,  
                verbose=2, shuffle=True,
                callbacks=[earlystopping],
                epochs=10_000, validation_split=0)
    rnn.save(filepath=f"./ProgrammingAssignment6/models/handwriting_rnn{n}.h5", overwrite=True, include_optimizer=True)
    with open(f"./ProgrammingAssignment6/logs/hist_handrnn{n}.pkl", mode="wb") as f:
        pickle.dump(log.history, f, protocol=pickle.HIGHEST_PROTOCOL)



    return None
if __name__ == "__main__":
    main()