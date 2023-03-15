# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fcnn import fcnn, fcnn_2layer

# %%
np.random.seed(6)

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
    np.random.shuffle(data)
    #return training_samples, test_samples
    return data[:train_sample_size, :], data[train_sample_size:train_sample_size+validation_sample_size, :], data[train_sample_size+validation_sample_size:, :]

# %% [markdown]
# # Regression on bivariate data

# %% [markdown]
# ### Importing data

# %%
file = '/Users/raunavghosh/Documents/DeepLearning/Assignments/ProgrammingAssignment2/Group10/Regression/BivariateData/10.csv'
df = pd.read_csv(file, header=None)
df.head()

# %%
data = df.to_numpy(dtype=float, copy=True)
data.shape

# %%
elivation, az_angle = 30, 60

# %%
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], edgecolor='black')
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
ax.set_title('Data given for regression task')

# %%
train, valid, test = train_test_split(data)
train.shape, valid.shape, test.shape

# %%
plt.figure()
ax = plt.subplot(3,1,1, projection='3d')
ax.scatter(train[:, 0], train[:, 1], train[:, 2], edgecolor='black')
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Training data for regression\n(Bivariate data)')

ax = plt.subplot(3,1,2, projection='3d')
ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2], edgecolor='black')
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Validation data for regression\n(Bivariate data)')

ax = plt.subplot(3,1,3, projection='3d')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], edgecolor='black')
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Testing data for regression\n(Bivariate data)')

plt.tight_layout(rect=[0, 0, 1, 3])


# %% [markdown]
# # Building the regressor

# %%
seed = 10
np.random.seed(seed)
neta = 0.02
max_epoch = 100
regressor = fcnn(node_layers=[2, 10, 1], max_epoch=max_epoch, learning_rate=neta, output_activation='linear')
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
for x in train[:, :-1]:
    pred.append(regressor.regress(x))

# %%
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(train[:, 0], train[:, 1], train[:, 2], edgecolor='black', label='Target')
ax.scatter(train[:, 0], train[:, 1], pred, marker='.', label='Model', edgecolors='red')
plt.legend()
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Model and Targeted data(Training)')
# plt.show()


# %%
plt.figure()
plt.scatter(train[:, 2], pred, label='True vs Model', edgecolors='black')
plt.plot([0,3], [0, 3], color='red', label='Matched line')
plt.title('Model vs Targeted(Training)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
# plt.show()

# %%
true_output = test[:, -1:]
pred_output = []
for point in test[:, :-1]:
    pred_output.append(regressor.regress(point))

# %%
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], edgecolor='black', label='Target')
ax.scatter(test[:, 0], test[:, 1], pred_output, marker='.', label='Model', edgecolors='red')
plt.legend()
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Model and Targeted data(Testing)')
# plt.show()

# %%
plt.figure()
plt.scatter(test[:, 2], pred_output, label='True vs Model', edgecolors='black')
plt.plot([0,3], [0, 3], color='red', label='Matched line')
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
train_output = train[:, -1:]
learned_output = [regressor.regress(point) for point in train[:, :-1]]
mse_train = mean_squared_error(train_output, learned_output)
mse_train

# %%
plt.figure()
plt.bar(['Training', 'Testing'], [mse_train, mse_test])
plt.title('Mean Squared Error')
plt.xlabel('Data')
plt.ylabel('Mean squared error')
# plt.show()

# %%
for node_number in range(10):
    node_class1 = []
    for sample in train:
        _, _, _, a = regressor.forward_propagate(sample[:-1])
        node_class1.append(a[node_number])
    plt.figure()    
    ax = plt.subplot(projection='3d')
    ax.scatter(train[:, 0], train[:, 1], node_class1)
    ax.set_zlabel('Activation value')
    ax.set_title(f'Hidden layer Node {node_number}')

# %% [markdown]
# # Building the regressor with 2 layer

# %%
seed = 10000
np.random.seed(seed)
neta = 0.01
max_epoch = 100
regressor = fcnn_2layer(node_layers=[2, 2, 2, 1], max_epoch=max_epoch, learning_rate=neta, output_activation='linear')
err_epoch, valid_err_epoch = regressor.fit_regressor(train, valid)

# %%
plt.figure()
plt.title("Error per epoch")
plt.plot(range(1, len(err_epoch)+1), err_epoch, label='Training loss')
plt.plot(range(1, len(err_epoch)+1), valid_err_epoch, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


# %% [markdown]
# # Validation

# %%
true_output = valid[:, -1:]
pred_output = []
for point in valid[:, :-1]:
    pred_output.append(regressor.regress(point))
mse_test = mean_squared_error(true_output, pred_output)
mse_test

# %%
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2], edgecolor='black', label='Target')
ax.scatter(valid[:, 0], valid[:, 1], pred_output, marker='.', label='Model', edgecolors='red')
plt.legend()
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Model and Targeted data(Validation)')
# plt.show()


# %%
pred = []
for x in train[:, :-1]:
    pred.append(regressor.regress(x))

# %%
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(train[:, 0], train[:, 1], train[:, 2], edgecolor='black', label='Target')
ax.scatter(train[:, 0], train[:, 1], pred, marker='.', label='Model', edgecolors='red')
plt.legend()
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Model and Targeted data(Training)')
# plt.show()


# %%
plt.figure()
plt.scatter(train[:, 2], pred, label='True vs Model', edgecolors='black')
plt.plot([0,3], [0, 3], color='red', label='Matched line')
plt.title('Model vs Targeted(Training)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
# plt.show()

# %%
true_output = test[:, -1:]
pred_output = []
for point in test[:, :-1]:
    pred_output.append(regressor.regress(point))

# %%
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], edgecolor='black', label='Target')
ax.scatter(test[:, 0], test[:, 1], pred_output, marker='.', label='Model', edgecolors='red')
plt.legend()
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.view_init(elivation, az_angle)
plt.title('Model and Targeted data(Testing)')
# plt.show()

# %%
plt.figure()
plt.scatter(test[:, 2], pred_output, label='True vs Model', edgecolors='black')
plt.plot([0,3], [0, 3], color='red', label='Matched line')
plt.title('Model vs Targeted(Testing)')
plt.xlabel('Target')
plt.ylabel('Model')
plt.legend()
# plt.show()

# %%
plt.figure()
mse_test = mean_squared_error(true_output, pred_output)
learned_output = [regressor.regress(point) for point in train[:, :-1]]
valid_output = [regressor.regress(point) for point in valid[:, :-1]]
mse_valid = mean_squared_error(valid[:, -1:], valid_output)
mse_train = mean_squared_error(train_output, learned_output)
mse_test, mse_valid, mse_train

# %%
plt.figure()
plt.bar(['Training', 'Validation', 'Testing'], [mse_train, mse_valid, mse_test])
plt.title('Mean Squared Error')
plt.xlabel('Data')
plt.ylabel('Mean squared error')
# plt.show()

# %% [markdown]
# # Hidden layer 1 outputs

# %%
for node_number in range(2):
    node_class1 = []
    for sample in train:
        _, _, _, _, _, a = regressor.forward_propagate(sample[:-1])
        node_class1.append(a[node_number])
    plt.figure()    
    ax = plt.subplot(projection='3d')
    ax.scatter(train[:, 0], train[:, 1], node_class1)
    ax.set_zlabel('Activation value')
    ax.set_title(f'Hidden layer Node {node_number}')

# %% [markdown]
# # Hidden layer 2 outputs

# %%
for node_number in range(2):
    node_class1 = []
    for sample in train:
        _, _, _, a, _, _ = regressor.forward_propagate(sample[:-1])
        node_class1.append(a[node_number])
    plt.figure()    
    ax = plt.subplot(projection='3d')
    ax.scatter(train[:, 0], train[:, 1], node_class1)
    ax.set_zlabel('Activation value')
    ax.set_title(f'Hidden layer Node {node_number}')

# %%
plt.show()


