# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fcnn import fcnn
from sklearn.metrics import mean_squared_error


# %%
np.random.seed(10)

# %%
def train_test_split(data):
    np.random.seed(10000)
    n_samples = data.shape[0]
    training_ratio = 0.6
    validation_ratio = 0.2
    # testing_ratio = 0.2
    train_sample_size = np.int_(n_samples*training_ratio)
    validation_sample_size = np.int_(n_samples*validation_ratio)
    # print(train_sample_size, validation_sample_size)
    np.random.shuffle(data)
    #return training_samples, test_samples
    return data[:train_sample_size, :], data[train_sample_size:train_sample_size+validation_sample_size, :], data[train_sample_size+validation_sample_size:, :]

# %% [markdown]
# # Regression on univariate data

# %% [markdown]
# ### Importing data

# %%
file_data = '/Users/raunavghosh/Documents/DeepLearning/Assignments/ProgrammingAssignment2/Group10/Regression/UnivariateData/10.csv'

# %%
df = pd.read_csv(file_data, header=None)
df.head()

# %%
data = df.to_numpy()

# %%
train, valid, test = train_test_split(data)
train.shape, valid.shape, test.shape

# %%
plt.figure()
plt.subplot(3, 1, 1)
plt.scatter(train[:, 0], train[:, 1], edgecolors='black')
plt.title('Training data')

plt.subplot(3, 1, 2)
plt.scatter(valid[:, 0], valid[:, 1], edgecolors='black')
plt.title('Validation data')

plt.subplot(3, 1, 3)
plt.scatter(test[:, 0], test[:, 1], edgecolors='black')
plt.title('Testing data')
plt.tight_layout(rect=[0, 0, 1.2, 3])
 

# %% [markdown]
# # Building the regressor

# %%
seed = 10
np.random.seed(seed)
neta = 0.02
max_epoch = 100
regressor = fcnn(node_layers=[1, 3, 1], max_epoch=max_epoch, learning_rate=neta, output_activation='linear')
epoch_err, valid_epoch_err = regressor.fit_regressor(train, valid)

# %%
plt.figure()
plt.title("Error per epoch")
plt.plot(range(1, len(epoch_err)+1), epoch_err, label='Training loss')
plt.plot(range(1, len(epoch_err)+1), valid_epoch_err, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


# %%
pred = []
for x in train[:, 0]:
    pred.append(regressor.regress([x]))

# %%
plt.figure()
plt.scatter(train[:, 0], train[:, 1], label='Training target', edgecolors='black')
# plt.title('Training data for regression (Univariate data)')
plt.scatter(train[:, 0], pred, label='Model', color='red', edgecolors='black', marker='^')
plt.title('Model and Targeted data(Training)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
 

# %%
plt.figure()
plt.scatter(train[:, 1], pred, label='True vs Model', edgecolors='black')
plt.plot([0, 3], [0, 3], color='red', label='Matched line')
plt.title('Model vs Targeted(Training)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
 

# %% [markdown]
# # Validation

# %%
true_output = valid[:, -1:]
pred_output = []
for point in valid[:, :-1].reshape((-1, 1)):
    pred_output.append(regressor.regress(point))

# %%
mse_valid = mean_squared_error(true_output, pred_output)
mse_valid

# %% [markdown]
# # Test

# %%
true_output = test[:, -1:]
pred_output = []
for point in test[:, :-1].reshape((-1, 1)):
    pred_output.append(regressor.regress(point))

# %%
plt.figure()
plt.scatter(test[:, 0], test[:, 1], label='Test target', edgecolors='black')
# plt.title('Training data for regression (Univariate data)')
plt.scatter(test[:, 0], pred_output, label='Model', color='red', edgecolors='black', marker='^')
plt.title('Model and Targeted data(Test)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
 

# %%
plt.figure()
plt.scatter(test[:, 1], pred_output, label='True vs Model', edgecolors='black')
plt.plot([0, 3], [0, 3], color='red', label='Matched line')
plt.title('Model vs Targeted(Testing)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
 

# %%
mse_test = mean_squared_error(true_output, pred_output)
mse_test

# %%
train_output = train[:, -1:]
learned_output = [regressor.regress(point) for point in train[:, :-1].reshape((-1, 1))]
mse_train = mean_squared_error(train_output, learned_output)
mse_train

# %%
plt.figure()
plt.bar(['Training', 'Validation', 'Testing'], [mse_train, mse_valid, mse_test])
plt.title('Mean Squared Error')
plt.xlabel('Data')
plt.ylabel('Mean squared error')
 

# %% [markdown]
# # Hidden layer output

# %%
for node_number in range(3):
    node_class1 = []
    for sample in train:
        _, _, _, a = regressor.forward_propagate(sample[:-1])
        node_class1.append(a[node_number])
    plt.figure()    
    ax = plt.subplot(projection='3d')
    ax.scatter(train[:, 0], train[:, 1], node_class1)
    ax.set_zlabel('Activation value')
    ax.set_title(f'Hidden layer Node {node_number}')

# %%



plt.show()