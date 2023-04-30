import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Resizing, Conv2D, MaxPooling2D, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from helper import data_import, relabel


def make_convnet():
    image = Input((224,224,3))
    #preprocessing layers
    x = Rescaling(1/255)(image)

    #conv block 1
    x = Conv2D(filters=8, kernel_size=(11,11), strides=4, padding="valid")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)

    #conv block 2
    x = Conv2D(filters=16, kernel_size=(5,5), strides=1, padding="valid")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)
    
    # FCNN block
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(5, activation="softmax")(x)

    return Model(inputs=[image], outputs=[x], name="Architecture1")


def main():
    train_path='./ProgrammingAssignment5/Group_10/train'
    test_path='./ProgrammingAssignment5/Group_10/test'
    val_path='./ProgrammingAssignment5/Group_10/val'

    # All of these are lists. Images are not of same shape, hence can't have np.array()
    x_train, y_train = data_import(train_path)
    x_test, y_test = data_import(test_path)
    x_val, y_val = data_import(val_path)

    y_train = relabel(y_train)
    y_test = relabel(y_test)
    y_val = relabel(y_val)

    # idx = np.random.randint(0,10)
    # plt.subplot(3,1,1)
    # plt.imshow(x_train[idx][:,:,::-1])

    # plt.subplot(3,1,2)
    # plt.imshow(x_test[idx][:,:,::-1])

    # plt.subplot(3,1,3)
    # plt.imshow(x_val[idx][:,:,::-1])

    # plt.show()

    model = make_convnet()
    model.summary()

    adam_optimizer = Adam(learning_rate = 0.001)
    model.compile(adam_optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
    history = model.fit(x=x_train, y=y_train,
                        batch_size=50, epochs=1000, # update on each class
                        callbacks=[earlystopping],
                        verbose=1, shuffle=True,
                        validation_split=0.0)
    
    model.save(filepath="./ProgrammingAssignment5/models/architecture1.h5", overwrite=True, include_optimizer=True)
    with open("./ProgrammingAssignment5/logs/hist_arch1.pkl", mode="wb") as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

