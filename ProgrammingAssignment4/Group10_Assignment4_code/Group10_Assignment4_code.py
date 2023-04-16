# %%
import numpy as np
from matplotlib import pyplot as plt
import pickle
from helper_fn import data_import, relabel

# %%
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# %%
from tensorflow.keras.models import load_model

# %%
test_path='./Group_10/test'
train_path='./Group_10/train'
val_path='./Group_10/val'

#import datasets
train_data, train_labels = data_import(train_path)
test_data, test_labels = data_import(test_path)
val_data, val_labels = data_import(val_path)

train_data = train_data.reshape(-1, 784)/255.0
test_data = test_data.reshape(-1, 784)/255.0
val_data = val_data.reshape(-1, 784)/255.0

train_labels = relabel(train_labels)
test_labels = relabel(test_labels)
val_labels = relabel(val_labels)

mean_vec = np.mean(train_data, axis=0) # 784 dimentional vector
# print(mean_vec.shape) 

# mean correction
train_data = train_data - mean_vec
test_data = test_data - mean_vec
val_data = val_data - mean_vec

# %%
np.unique(train_labels)

# %%
train_data.shape, test_data.shape, val_data.shape

# %% [markdown]
# # PCA Task1

# %% [markdown]
# ## 32 PCA

# %%
with open("./pca_models/pca32.pkl", mode="rb") as f:
    pca32 = pickle.load(f)

reduced_train = pca32.transform(train_data)
reduced_test = pca32.transform(test_data)
reduced_val = pca32.transform(val_data)

# %%
model = load_model("./pca_models/pcamodel1_32.h5")
model.summary()
print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))



# %%
model = load_model("./pca_models/pcamodel2_32.h5")
model.summary()
print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))


# %% [markdown]
# ## 64 PCA

# %%
with open("./pca_models/pca64.pkl", mode="rb") as f:
    pca64 = pickle.load(f)

reduced_train = pca64.transform(train_data)
reduced_test = pca64.transform(test_data)
reduced_val = pca64.transform(val_data)

# %%
model = load_model("./pca_models/pcamodel1_64.h5")
model.summary()
print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))



# %%
model = load_model("./pca_models/pcamodel2_64.h5")
model.summary()
print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))

# %% [markdown]
# ## 128 PCA

# %%
with open("./pca_models/pca128.pkl", mode="rb") as f:
    pca128 = pickle.load(f)

reduced_train = pca128.transform(train_data)
reduced_test = pca128.transform(test_data)
reduced_val = pca128.transform(val_data)

# %%
model = load_model("./pca_models/pcamodel1_128.h5")
model.summary()

print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))


# %%
model = load_model("./pca_models/pcamodel2_128.h5")
model.summary()

print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))


# %% [markdown]
# ## 256 PCA

# %%
with open("./pca_models/pca256.pkl", mode="rb") as f:
    pca256 = pickle.load(f)

reduced_train = pca256.transform(train_data)
reduced_test = pca256.transform(test_data)
reduced_val = pca256.transform(val_data)

# %%
model = load_model("./pca_models/pcamodel1_256.h5")
model.summary()

print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))


# %%
model = load_model("./pca_models/pcamodel2_256.h5")
model.summary()

print("Train data eval:")
print(model.evaluate(reduced_train, train_labels))

print("Test data eval:")
print(model.evaluate(reduced_test, test_labels))

print("Val data eval:")
print(model.evaluate(reduced_val, val_labels))


# %% [markdown]
# ## Best architechture [64,256,128,64,32,5]

# %%
from tensorflow.math import confusion_matrix

# %%
with open("./pca_models/pca64.pkl", mode="rb") as f:
    pca64 = pickle.load(f)

reduced_train = pca64.transform(train_data)
reduced_test = pca64.transform(test_data)
reduced_val = pca64.transform(val_data)

model = load_model("./pca_models/pcamodel2_64.h5")
model.summary()
# print("Train data eval:")
# print(model.evaluate(reduced_train, train_labels))

# print("Test data eval:")
# print(model.evaluate(reduced_test, test_labels))

# print("Val data eval:")
# print(model.evaluate(reduced_val, val_labels))

pred_test = model.predict(reduced_test)
pred_labels = [np.argmax(label) for label in pred_test]
print(confusion_matrix(test_labels, pred_labels).numpy())

# %% [markdown]
# # Autoencoders

# %%
test_path='./Group_10/test'
train_path='./Group_10/train'
val_path='./Group_10/val'

#import datasets
train_data, train_labels = data_import(train_path)
test_data, test_labels = data_import(test_path)
val_data, val_labels = data_import(val_path)

train_data = train_data.reshape(-1, 784)/255.0
test_data = test_data.reshape(-1, 784)/255.0
val_data = val_data.reshape(-1, 784)/255.0

train_labels = relabel(train_labels)
test_labels = relabel(test_labels)
val_labels = relabel(val_labels)

# mean_vec = np.mean(train_data, axis=0) # 784 dimentional vector
# print(mean_vec.shape) 

# %% [markdown]
# ## One layer 32 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_1layer_32.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_32_1layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_32_1layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_32_1layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_1layer_32.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(32,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{32}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(32,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{32}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## Three layer 32 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_3layer_32.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_32_3layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_32_3layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_32_3layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_3layer_32.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(32,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{32}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(32,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{32}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## One layer 64 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_1layer_64.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_64_1layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_64_1layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_64_1layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_1layer_64.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(64,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{64}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(64,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{64}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## Three layer 64 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_3layer_64.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_64_3layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_64_3layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_64_3layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_3layer_64.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(64,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{64}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(64,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{64}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## One layer 128 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_1layer_128.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_128_1layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_128_1layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_128_1layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_1layer_128.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(128,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{128}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(128,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{128}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## Three layer 128 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_3layer_128.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_128_3layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_128_3layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_128_3layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_3layer_128.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(128,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{128}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(128,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{128}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## One layer 256 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_1layer_256.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_256_1layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_256_1layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_256_1layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_1layer_256.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(256,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{256}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(256,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{256}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %%
from tensorflow.math import confusion_matrix
pred_test = model.predict(encoded_test)
pred_labels = [np.argmax(label) for label in pred_test]
print(confusion_matrix(test_labels, pred_labels).numpy())

# %% [markdown]
# ## Three layer 256 bottleneck

# %%
model = load_model("./autoencoder_models/autoencoder_3layer_256.h5")
model.summary()

# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_256_3layer.png')


# %%
class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('train_recon_256_3layer.png')


# %%
class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('test_recon_256_3layer.png')


# %%
class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
# model.summary()

plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')
plt.savefig('val_recon_256_3layer.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### Model 1

# %%
encoder = load_model("./autoencoder_models/encoder_3layer_256.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(256,), name="Input")
x = Dense(16, activation='tanh', name="Layer1")(inputs)
x = Dense(8, activation='tanh', name="Layer2")(x)
# x = Dense(128, activation='tanh', name="Layer3")(x)
# x = Dense(64, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{256}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                min_delta=1e-4,
                                patience=5,
                                verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ### Model 2

# %%
inputs = Input(shape=(256,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{256}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# # Denoising Autoencoder

# %% [markdown]
# ## 20% noise

# %%
model = load_model("./denoising-autoencoder_models/autoencoder_1layer_0.2.h5")


# %%
model.summary()

class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]
plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')

plt.suptitle("Training")
plt.savefig('train_denoising_recon.png')

# %%
model.summary()

class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')

plt.suptitle("Testing")
plt.savefig('test_denoising_recon.png')


# %%
model.summary()

class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')

plt.suptitle("Validation")
plt.savefig('val_denoising_recon.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### FCNN

# %%
encoder = load_model("./denoising-autoencoder_models/encoder_1layer_0.2.h5")
encoded_train = encoder.predict(train_data)
encoded_test = encoder.predict(test_data)
encoded_val = encoder.predict(val_data)

# %%
inputs = Input(shape=(128,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{128}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# ## 40% noise

# %%
model = load_model("./denoising-autoencoder_models/autoencoder_1layer_0.4.h5")


# %%
model.summary()

class1 = train_data[0+100, :]
class2 = train_data[2277+100, :]
class3 = train_data[2*2277+100, :]
class4 = train_data[3*2277+100, :]
class5 = train_data[4*2277+100, :]
plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')

plt.suptitle("Training")
plt.savefig('train_denoising_recon_0.4.png')


# %%
model.summary()

class1 = test_data[0+100, :]
class2 = test_data[759+100, :]
class3 = test_data[2*759+100, :]
class4 = test_data[3*759+100, :]
class5 = test_data[4*759+100, :]
plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')

plt.suptitle("Testing")
plt.savefig('test_denoising_recon_0.4.png')


# %%
model.summary()

class1 = val_data[0+100, :]
class2 = val_data[759+100, :]
class3 = val_data[2*759+100, :]
class4 = val_data[3*759+100, :]
class5 = val_data[4*759+100, :]
plt.figure()
plt.subplot(2,5,1)
# plt.title("Original")
plt.imshow(class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,2)
plt.imshow(class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,3)
plt.imshow(class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,4)
plt.imshow(class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,5)
plt.imshow(class5.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,6)
# plt.title("Original")
recon_class1 = model.predict(class1.reshape((1, -1)))
plt.imshow(recon_class1.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,7)
recon_class2 = model.predict(class2.reshape((1, -1)))
plt.imshow(recon_class2.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,8)
recon_class3 = model.predict(class3.reshape((1, -1)))
plt.imshow(recon_class3.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,9)
recon_class4 = model.predict(class4.reshape((1, -1)))
plt.imshow(recon_class4.reshape((28,28)))
plt.axis('off')

plt.subplot(2,5,10)
recon_class5 = model.predict(class5.reshape((1, -1)))
plt.imshow(recon_class5.reshape((28,28)))
plt.axis('off')

plt.suptitle("Validation")
plt.savefig('val_denoising_recon_0.4.png')


# %%
train_pred = model.predict(train_data)
test_pred = model.predict(test_data)
val_pred = model.predict(val_data)

print("Reconstrunction error")
print("Training: ", np.average(((train_data - train_pred)**2)))
print("Testing: ", np.average(((test_data - test_pred)**2)))
print("Validation: ", np.average(((val_data - val_pred)**2)))


# %% [markdown]
# ### FCNN

# %%
inputs = Input(shape=(128,), name="Input")
x = Dense(256, activation='tanh', name="Layer1")(inputs)
x = Dense(128, activation='tanh', name="Layer2")(x)
x = Dense(64, activation='tanh', name="Layer3")(x)
x = Dense(32, activation='tanh', name="Layer4")(x)
outputs = Dense(5, activation='softmax', name="Output")(x)
model = Model(inputs=inputs, outputs=outputs, name=f"Model-encoder{128}")
model.summary()

adam_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer=adam_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5,
                                        verbose=1)
history1 = model.fit(x=encoded_train, y=train_labels,
                        batch_size=32, epochs=100_000,
                        callbacks=[earlystopping],
                        verbose=0, shuffle=True,
                        validation_split=0.0)

# %%
print("Training data evaluation")
model.evaluate(encoded_train, train_labels)
print("Testing data evaluation")
model.evaluate(encoded_test, test_labels)
print("Validation data evaluation")
model.evaluate(encoded_val, val_labels)
print()

# %% [markdown]
# # Visualizing weights

# %% [markdown]
# ## Autoencoder

# %%
model = load_model("./autoencoder_models/autoencoder_1layer_128.h5")

# %%
weights = model.layers[1].get_weights()
weights[0].shape

# %%
w31 = []

for w in weights[0].T:
    w31.append(w.reshape(28, 28))
    
w31 = np.array(w31)
w31.shape

# %%
n = 9
plt.figure(figsize=(10, 10))
for i in range(n):
    k = np.random.randint(0, 128)
    ax = plt.subplot(3, 3, i+1)
    plt.title("Node no: "+str(k))
    plt.imshow(w31[k])
    plt.axis('off')
plt.savefig('weights-autoencoder.png', bbox_inches='tight')
#plt.show()

# %% [markdown]
# ## Denoising autoencoder

# %% [markdown]
# ### 20% noise

# %%
model = load_model("./denoising-autoencoder_models/autoencoder_1layer_0.2.h5")

# %%
weights = model.layers[1].get_weights()
weights[0].shape

# %%
w31 = []

for w in weights[0].T:
    w31.append(w.reshape(28, 28))
    
w31 = np.array(w31)
w31.shape

# %%
n = 9
plt.figure(figsize=(10, 10))
for i in range(n):
    k = np.random.randint(0, 128)
    ax = plt.subplot(3, 3, i+1)
    plt.title("Node no: "+str(k))
    plt.imshow(w31[k])
    plt.axis('off')
plt.savefig('weights-denoiseing-0.2.png', bbox_inches='tight')
#plt.show()

# %% [markdown]
# ### 40% noise

# %%
model = load_model("./denoising-autoencoder_models/autoencoder_1layer_0.4.h5")

# %%
weights = model.layers[1].get_weights()
weights[0].shape

# %%
w31 = []

for w in weights[0].T:
    w31.append(w.reshape(28, 28))
    
w31 = np.array(w31)
w31.shape

# %%
n = 9
plt.figure(figsize=(10, 10))
for i in range(n):
    k = np.random.randint(0, 128)
    ax = plt.subplot(3, 3, i+1)
    plt.title("Node no: "+str(k))
    plt.imshow(w31[k])
    plt.axis('off')
plt.savefig('weights-denoiseing-0.4.png', bbox_inches='tight')
#plt.show()

# %%



plt.show()