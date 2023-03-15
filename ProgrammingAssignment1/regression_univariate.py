# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from perceptron_v2 import perceptron

# %%
def train_test_split(data, training_ratio=0.7):
    train_sample_size = np.int_(data.shape[0]*training_ratio)
    np.random.shuffle(data)
    #return training_samples, test_samples
    return data[:train_sample_size, :], data[train_sample_size:, :]

# %% [markdown]
# # Regression for Univariate data

# %%
file = '/Users/raunavghosh/Documents/DeepLearning/Assignments/ProgrammingAssignment1/Group10/Regression/UnivariateData/10.csv'

# %%
df = pd.read_csv(file, header=None)
df.head()

# %%
data = df.to_numpy(dtype=float, copy=True)
data.shape

# %%
plt.figure()
plt.scatter(data[:, 0], data[:, 1], edgecolors='black')
plt.title('Data given for regression task')
# plt.show()

# %%
train_data, test_data = train_test_split(data)
train_data.shape, test_data.shape

# %%
plt.figure(figsize=(15, 7))
plt.subplot(1,2,1)
plt.scatter(train_data[:, 0], train_data[:, 1], edgecolors='black')
plt.title('Training data for regression (Univariate data)')

plt.subplot(1,2,2)
plt.scatter(test_data[:, 0], test_data[:, 1], edgecolors='black')
plt.title('Testing data for regression (Univariate data)')
# plt.tight_layout(rect=[0, 0, 2, 1.2])
# plt.show()

# %% [markdown]
# # Building the Regressor

# %%
neta = 0.005
regressor = perceptron(n_features=1, activation='linear', learning_rate=neta, max_epoch=100)
epoch_err = regressor.fit_regression(train_data)

# %%
plt.figure()
plt.title("MSE per epoch")
plt.plot(range(1, len(epoch_err)+1), epoch_err)
plt.xlabel('Epoch')
plt.ylabel('MSE')
# plt.show()

# %%
pred = []
for x in train_data[:, 0]:
    pred.append(regressor.predict_regress([x]))

# %%
plt.figure()
plt.scatter(train_data[:, 0], train_data[:, 1], label='Training target', edgecolors='black')
# plt.title('Training data for regression (Univariate data)')
plt.scatter(train_data[:, 0], pred, label='Model', color='red', edgecolors='black', marker='^')
plt.title('Model and Targeted data(Training)')
plt.legend()
# plt.show()

# %%
plt.figure()
plt.scatter(train_data[:, 1], pred, label='True vs Model', edgecolors='black')
plt.plot([0,2], [0, 2], color='red', label='Matched line')
plt.title('Model vs Targeted(Training)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
# plt.show()

# %% [markdown]
# ## Testing

# %%
true_output = test_data[:, -1:]
pred_output = []
for point in test_data[:, :-1].reshape((-1, 1)):
    pred_output.append(regressor.predict_regress(point))

# %%
plt.figure()
plt.scatter(test_data[:, 0], test_data[:, 1], label='Test target', edgecolors='black')
# plt.title('Training data for regression (Univariate data)')
plt.scatter(test_data[:, 0], pred_output, label='Model', color='red', edgecolors='black', marker='^')
plt.title('Model and Targeted data(Test)')
plt.legend()
# plt.show()

# %%
plt.figure()
plt.scatter(test_data[:, 1], pred_output, label='True vs Model', edgecolors='black')
plt.plot([0,2], [0, 2], color='red', label='Matched line')
plt.title('Model vs Targeted(Testing)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
# plt.show()

# %%
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(true_output, pred_output)
mse_test

# %%
train_output = train_data[:, -1:]
learned_output = [regressor.predict_regress(point) for point in train_data[:, :-1].reshape((-1, 1))]
mse_train = mean_squared_error(train_output, learned_output)
mse_train

# %%
plt.figure()
plt.bar(['Training', 'Testing'], [mse_train, mse_test])
plt.title('Mean Squared Error')
plt.xlabel('Data')
plt.ylabel('Mean squared error')
plt.show()

# %%



