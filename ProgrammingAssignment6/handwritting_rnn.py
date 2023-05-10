import tensorflow as tf
from tensorflow.keras import Model, Input, layers, optimizers, callbacks, Sequential, losses
import numpy as np
import matplotlib.pyplot as plt
import pickle

from preprocessing import process_handwriting, data_import_handwriting, StoppingCriteria

# def padding_sequences(dataset, mask_val=np.nan):
#     sequence_lengths = [len(seq) for seq in dataset] # length of all the sequences in the dataset
#     max_len = max(sequence_lengths)

#     dataset_padded = []
#     for seq, seq_len in zip(dataset, sequence_lengths):
#         seq_padded = [x for x in seq] # copy the sequence
#         pad_len = max_len - seq_len
#         for i in range(pad_len):
#             seq_padded.append(mask_val) # pad
#         # print(seq_padded)
#         dataset_padded.append(seq_padded)
#     return tf.expand_dims(dataset_padded, axis=-1), max_len


# def make_rnn(mask_val=10_000):
#     """
#     Following the example from
#     https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM#what-are-many-to-one-sequence-problems?
#     """
#     # timesteps = 200
#     features = 2
#     # inputs = Input(shape=(200,2))
#     # x = layers.Masking(mask_value=mask_val)(inputs)
#     # x = layers.SimpleRNN(units=10, activation="tanh", input_shape=(200,2))(x)
#     # outputs = layers.Dense(units=5, activation="Softmax")(x)

#     # # # raise NotImplementedError
#     # return Model(inputs=[inputs,], outputs=[outputs,], name="Handwriting-RNN")
#     model = Sequential()
#     model.add(layers.Masking(mask_value=mask_val, input_shape=(None, features)))
#     model.add(layers.SimpleRNN(10, activation='tanh'))
#     model.add(layers.Dense(5, activation="Softmax"))
#     return model


# def main():
#     tf.random.set_seed(32)
#     data_path = "./ProgrammingAssignment6/CS671-DLA-Assignment4-Data-2022/Handwriting_Data"
#     x_train, y_train, x_test, y_test = data_import_handwriting(data_path)
#     # print(x_train[3].shape)
#     # mask_val=np.nan 
#     """
#     Can't use np.nan as mask value since (np.nan == np.nan)= False
#     """
#     mask_val = np.nan
#     # x_train, x_max = padding_sequences(x_train)
#     x_train = tf.keras.utils.pad_sequences(x_train, value=mask_val, padding="post", dtype=np.float64, maxlen=200)
#     # print(x_train[0])
#     y_train = np.array(y_train)
#     # print(y_train)
#     # x_test = padding_sequences(x_test)

#     rnn = make_rnn(mask_val=mask_val)
#     # print(rnn(x_train[0]))
#     rnn.summary()
#     adam_optimizer = optimizers.Adam()
#     rnn.compile(adam_optimizer, 
#                 loss="sparse_categorical_crossentropy",
#                 metrics=['accuracy'])
#     earlystopping = callbacks.EarlyStopping(monitor='loss',
#                                         min_delta=1e-4,
#                                         patience=3,
#                                         verbose=1)
#     # print(rnn.predict(np.expand_dims(x_train[0], 0)))
#     # print(x_train.shape)
#     history = rnn.fit(x=x_train, y=y_train,
#                     batch_size=1, epochs=10_000, # update on each example
#                     callbacks=[earlystopping],
#                     verbose=2, shuffle=False,
#                     validation_split=0.0)
#     rnn.save(filepath="./ProgrammingAssignment6/models/handwriting_rnn1.h5", overwrite=True, include_optimizer=True)
#     with open("./ProgrammingAssignment6/logs/hist_handrnn1.pkl", mode="wb") as f:
#         pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
# def make_rnn():
#     features=2
#     seq_len = 200 # fixing for the sake of sanity
#     model = Sequential()
#     model.add(layers.Masking(mask_value=-1, input_shape=(seq_len, features)))
#     model.add(layers.SimpleRNN(5, input_shape=(seq_len, features)))
#     model.add(layers.Dense(5, activation="Softmax"))
#     return model
def make_rnn2():
    features=2
    seq_len = 200 # fixing for the sake of sanity
    model = Sequential()
    model.add(layers.Masking(mask_value=-1, input_shape=(None, features)))
    model.add(layers.SimpleRNN(10, return_sequences=True))
    model.add(layers.SimpleRNN(5))
    model.add(layers.Dense(5, activation="Softmax"))
    return model
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
    n = 2 # model number
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
    # rnn = make_rnn()
    rnn = make_rnn2()

    rnn.summary()
    adam_optimizer = optimizers.Adam(learning_rate=0.001)
    categorical_loss = losses.SparseCategoricalCrossentropy()
    rnn.compile(optimizer=adam_optimizer, loss=categorical_loss, metrics=["accuracy"])
    earlystopping = callbacks.EarlyStopping(monitor='loss',
                                            min_delta=0.0001,
                                            patience=10,
                                            verbose=1)
    terminate_on_nan = callbacks.TerminateOnNaN()
    # stopping_criteria = StoppingCriteria(patience=1, min_delta=1e-4)
    log = rnn.fit(x=x_train_padded, y=y_train,  
                verbose=2, shuffle=True,
                callbacks=[terminate_on_nan],
                epochs=10_000, validation_split=0)
    rnn.save(filepath=f"./ProgrammingAssignment6/models/handwriting_rnn{n}.h5", overwrite=True, include_optimizer=True)
    with open(f"./ProgrammingAssignment6/logs/hist_handrnn{n}.pkl", mode="wb") as f:
        pickle.dump(log.history, f, protocol=pickle.HIGHEST_PROTOCOL)



    return None
if __name__ == "__main__":
    main()