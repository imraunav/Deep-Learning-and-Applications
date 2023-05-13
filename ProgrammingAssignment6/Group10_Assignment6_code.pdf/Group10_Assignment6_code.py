# %%
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from matplotlib import pyplot as plt
from preprocessing import process_handwriting, data_import_handwriting
import numpy as np
from tensorflow.math import confusion_matrix


# %%
data_path = "./CS671-DLA-Assignment4-Data-2022/Handwriting_Data"
x_train, y_train, x_test, y_test = data_import_handwriting(data_path)
# print(x_train[3].shape)
x_max=np.nan
# x_train, x_max = padding_sequences(x_train)
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)
# print(y_train)
# x_test = padding_sequences(x_test)


# %%
import matplotlib.pyplot as plt
# rand_idx = np.random.randint(0,344,(25,))
# plt.figure(figsize=(10,10))
# for i, idx in enumerate(rand_idx):
#     plt.subplot(5,5,i+1)
#     plt.plot(x_train[idx][:, 0], x_train[idx][:, 1])
# plt.tight_layout()
# plt.show()
# a, bA, chA, lA, tA
# 0-68, 69-135, 136-205, 206-273, 274-342

# %%
# rand_idx = np.random.randint(0,344,(25,))
rand_idx_a = np.random.randint(0, 68, (5,))
rand_idx_bA = np.random.randint(68, 135, (5,))
rand_idx_chA = np.random.randint(136, 205, (5,))
rand_idx_lA = np.random.randint(206, 273, (5,))
rand_idx_tA = np.random.randint(274, 342, (5,))
rand_idx = [rand_idx_a, rand_idx_bA, rand_idx_chA, rand_idx_lA, rand_idx_tA]

for idx_set in rand_idx:
    plt.figure(figsize=(9, 2))
    for i, idx in enumerate(idx_set):
        plt.subplot(1,5,i+1)
        plt.plot(x_train[idx][:, 0], x_train[idx][:, 1])
    plt.tight_layout()
    plt.show()
    


# %% [markdown]
# # Handwriting

# %% [markdown]
# ## RNN-Architecture1

# %%
rnn = load_model("./models/handwriting_rnn1.h5")
rnn.summary()

# %%
with open("./logs/hist_handrnn1.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 1")
plt.tight_layout()
plt.savefig("architecture1.png")

# %%

mask_val=-1
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## RNN-Architecture2

# %%
rnn = load_model("./models/handwriting_rnn2.h5")
rnn.summary()

# %%
with open("./logs/hist_handrnn2.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 2")
plt.tight_layout()
plt.savefig("architecture2.png")

# %%

mask_val=-1
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## RNN-Architecture3

# %%
rnn = load_model("./models/handwriting_rnn3.h5")
rnn.summary()

# %%
with open("./logs/hist_handrnn3.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 3")
plt.tight_layout()
plt.savefig("architecture3.png")

# %%

mask_val=-1
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## LSTM-Architecture1

# %%
rnn = load_model("./models/handwriting_lstm1.h5")
rnn.summary()

# %%
with open("./logs/hist_handlstm1.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 1")
plt.tight_layout()
plt.savefig("lstm-architecture1.png")

# %%

mask_val=-1
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## LSTM-Architecture2

# %%
rnn = load_model("./models/handwriting_lstm2.h5")
rnn.summary()

# %%
with open("./logs/hist_handlstm2.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 2")
plt.tight_layout()
plt.savefig("lstm-architecture2.png")

# %%

mask_val=-1
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %%


# %% [markdown]
# ## LSTM-Architecture3

# %%
rnn = load_model("./models/handwriting_lstm3.h5")
rnn.summary()

# %%
with open("./logs/hist_handlstm3.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 3")
plt.tight_layout()
plt.savefig("lstm-architecture3.png")

# %%

mask_val=-1
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=200)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=200)

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# # CV

# %%
from preprocessing import data_import_cv

# %%
x_train, y_train, x_test, y_test = data_import_cv("./CS671-DLA-Assignment4-Data-2022/CV_Data/")
mask_val=100_000.0
x_train_padded = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_val, maxlen=79)
x_test_padded = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_val, maxlen=79)

y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

# %%
x_train_padded[0].shape

# %% [markdown]
# ## RNN-Architecture1

# %%
rnn = load_model("./models/cv_rnn1.h5")
rnn.summary()

# %%
with open("./logs/hist_cvrnn1.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 1")
plt.tight_layout()
plt.savefig("cv-architecture1.png")

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## RNN-Architecture2

# %%
rnn = load_model("./models/cv_rnn2.h5")
rnn.summary()

# %%
with open("./logs/hist_cvrnn2.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 2")
plt.tight_layout()
plt.savefig("cv-architecture2.png")

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## RNN-Architecture3

# %%
rnn = load_model("./models/cv_rnn3.h5")
rnn.summary()

# %%
with open("./logs/hist_cvrnn3.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 3")
plt.tight_layout()
plt.savefig("cv-architecture3.png")

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## LSTM-Architecture1

# %%
rnn = load_model("./models/cv_lstm1.h5")
rnn.summary()

# %%
with open("./logs/hist_cvlstm1.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 1")
plt.tight_layout()
plt.savefig("lstm-cv-architecture1.png")

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %% [markdown]
# ## LSTM-Architecture2

# %%
rnn = load_model("./models/cv_lstm2.h5")
rnn.summary()

# %%
with open("./logs/hist_cvlstm2.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 2")
plt.tight_layout()
plt.savefig("lstm-cv-architecture2.png")

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %%


# %% [markdown]
# ## LSTM-Architecture3

# %%
rnn = load_model("./models/cv_lstm3.h5")
rnn.summary()

# %%
with open("./logs/hist_cvlstm3.pkl", mode="rb") as file:
    hist = pickle.load(file)
plt.subplot(2,1,1)
plt.plot(hist["loss"])
plt.xlabel("Epochs")
plt.title("Loss")

plt.subplot(2,1,2)
plt.plot(hist["accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.suptitle("Architecture 3")
plt.tight_layout()
plt.savefig("lstm-cv-architecture3.png")

# %%
y_pdfs = rnn.predict(x_train_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_train, y_pred).numpy())

# %%
rnn.evaluate(x_train_padded, y_train)

# %%
rnn.evaluate(x_test_padded, y_test)


# %%
y_pdfs = rnn.predict(x_test_padded)
y_pred = tf.argmax(y_pdfs, axis=1)
print(confusion_matrix(y_test, y_pred).numpy())

# %%


# %%



