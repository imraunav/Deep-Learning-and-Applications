import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from helper import data_import, relabel

def main():
    # Base model
    vgg19 = VGG19(input_shape=(224,224,3),weights="imagenet", include_top=False)

    # vgg19.summary()

    # fcnn at the end of the network
    flat = layers.Flatten()(vgg19.output)
    x = layers.Dense(5, activation="softmax")(flat)

    model = Model(inputs=vgg19.input,
                outputs=x,
                name='Modified-VGG19')

    # Freezing the base model weights
    for layer in vgg19.layers:
        layer.trainable=False

    model.summary()

    # importing training data
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
    
    model.save(filepath="./ProgrammingAssignment5/models/vgg19_mod.h5", overwrite=True, include_optimizer=True)
    with open("./ProgrammingAssignment5/logs/hist_vgg19_mod.pkl", mode="wb") as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()