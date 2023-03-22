# %%
from tensorflow import keras
import pickle
from matplotlib import pyplot as plt

# %%
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD_history1.pkl",
          mode= 'rb') as f:
    sgd_history1 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/batch_history1.pkl",
          mode= 'rb') as f:
    batch_history1 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD-moment_history1.pkl",
          mode= 'rb') as f:
    sgd_moment_history1 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/NAG_history1.pkl",
          mode= 'rb') as f:
    nag_history1 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/adagrad_history1.pkl",
          mode= 'rb') as f:
    adagrad_history1 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/rmsprop_history1.pkl",
          mode= 'rb') as f:
    rmsprop_history1 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/Adam_history1.pkl",
          mode= 'rb') as f:
    adam_history1 = pickle.load(f)

# %%
plt.plot(sgd_history1['loss'], label='SGD')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-sgd.png')

plt.clf()

plt.plot(batch_history1['loss'], label='Batch')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-batch.png')

plt.clf()

plt.plot(sgd_moment_history1['loss'], label='SGD with momentum')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-sgd-moment.png')

plt.clf()

plt.plot(nag_history1['loss'], label='NAG')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-nag.png')

plt.clf()

plt.plot(adagrad_history1['loss'], label='AdaGrad')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-adagrad.png')

plt.clf()

plt.plot(rmsprop_history1['loss'], label='RMSProp')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-rmsprop.png')

plt.clf()

plt.plot(adam_history1['loss'], label='Adam')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch1-adam.png')

plt.clf()


# %%
print(len(sgd_history4['loss']),
len(batch_history4['loss']),
len(sgd_moment_history4['loss']),
len(nag_history4['loss']),
len(adagrad_history4['loss']),
len(rmsprop_history4['loss']),
len(adam_history4['loss']),
)

# %%
plt.plot(sgd_history1['loss'], label='SGD')
plt.plot(batch_history1['loss'], label='Batch')
plt.plot(sgd_moment_history1['loss'], label='SGD with momentum')
plt.plot(nag_history1['loss'], label='NAG')
plt.plot(adagrad_history1['loss'], label='AdaGrad')
plt.plot(rmsprop_history1['loss'], label='RMSProp')
plt.plot(adam_history1['loss'], label='Adam')

plt.title("Architechture 1")
# plt.xlim((0, 15))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Arch1-super.png')

# %%
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD_history2.pkl",
          mode= 'rb') as f:
    sgd_history2 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/batch_history2.pkl",
          mode= 'rb') as f:
    batch_history2 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD-moment_history2.pkl",
          mode= 'rb') as f:
    sgd_moment_history2 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/NAG_history2.pkl",
          mode= 'rb') as f:
    nag_history2 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/adagrad_history2.pkl",
          mode= 'rb') as f:
    adagrad_history2 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/rmsprop_history2.pkl",
          mode= 'rb') as f:
    rmsprop_history2 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/Adam_history2.pkl",
          mode= 'rb') as f:
    adam_history2 = pickle.load(f)

# %%
plt.plot(sgd_history2['loss'], label='SGD')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-sgd.png')

plt.clf()

plt.plot(batch_history2['loss'], label='Batch')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-batch.png')

plt.clf()

plt.plot(sgd_moment_history2['loss'], label='SGD with momentum')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-sgd-moment.png')

plt.clf()

plt.plot(nag_history2['loss'], label='NAG')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-nag.png')

plt.clf()

plt.plot(adagrad_history2['loss'], label='AdaGrad')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-adagrad.png')

plt.clf()

plt.plot(rmsprop_history2['loss'], label='RMSProp')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-rmsprop.png')

plt.clf()

plt.plot(adam_history2['loss'], label='Adam')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch2-adam.png')

plt.clf()


# %%
plt.plot(sgd_history2['loss'], label='SGD')
plt.plot(batch_history2['loss'], label='Batch')
plt.plot(sgd_moment_history2['loss'], label='SGD with momentum')
plt.plot(nag_history2['loss'], label='NAG')
plt.plot(adagrad_history2['loss'], label='AdaGrad')
plt.plot(rmsprop_history2['loss'], label='RMSProp')
plt.plot(adam_history2['loss'], label='Adam')
plt.title("Architechture 2")

# plt.xlim((0,15))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Arch2-super.png")

# %%
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD_history3.pkl",
          mode= 'rb') as f:
    sgd_history3 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/batch_history3.pkl",
          mode= 'rb') as f:
    batch_history3 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD-moment_history3.pkl",
          mode= 'rb') as f:
    sgd_moment_history3 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/NAG_history3.pkl",
          mode= 'rb') as f:
    nag_history3 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/adagrad_history3.pkl",
          mode= 'rb') as f:
    adagrad_history3 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/rmsprop_history3.pkl",
          mode= 'rb') as f:
    rmsprop_history3 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/Adam_history3.pkl",
          mode= 'rb') as f:
    adam_history3 = pickle.load(f)

# %%
plt.plot(sgd_history3['loss'], label='SGD')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-sgd.png')

plt.clf()

plt.plot(batch_history3['loss'], label='Batch')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-batch.png')

plt.clf()

plt.plot(sgd_moment_history3['loss'], label='SGD with momentum')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-sgd-moment.png')

plt.clf()

plt.plot(nag_history3['loss'], label='NAG')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-nag.png')

plt.clf()

plt.plot(adagrad_history3['loss'], label='AdaGrad')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-adagrad.png')

plt.clf()

plt.plot(rmsprop_history3['loss'], label='RMSProp')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-rmsprop.png')

plt.clf()

plt.plot(adam_history3['loss'], label='Adam')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch3-adam.png')

plt.clf()


# %%
plt.plot(sgd_history3['loss'], label='SGD')
plt.plot(batch_history3['loss'], label='Batch')
plt.plot(sgd_moment_history3['loss'], label='SGD with momentum')
plt.plot(nag_history3['loss'], label='NAG')
plt.plot(adagrad_history3['loss'], label='AdaGrad')
plt.plot(rmsprop_history3['loss'], label='RMSProp')
plt.plot(adam_history3['loss'], label='Adam')

plt.title('Architecture 3')
# plt.xlim((0,15))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Arch3-super.png")

# %%
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD_history4.pkl",
          mode= 'rb') as f:
    sgd_history4 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/batch_history4.pkl",
          mode= 'rb') as f:
    batch_history4 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/SGD-moment_history4.pkl",
          mode= 'rb') as f:
    sgd_moment_history4 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/NAG_history4.pkl",
          mode= 'rb') as f:
    nag_history4 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/adagrad_history4.pkl",
          mode= 'rb') as f:
    adagrad_history4 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/rmsprop_history4.pkl",
          mode= 'rb') as f:
    rmsprop_history4 = pickle.load(f)
with open("/Users/raunavghosh/Documents/DeepLearning/Assignments/Deep-Learning-and-Applications-CS671/ProgrammingAssignment3/Adam_history4.pkl",
          mode= 'rb') as f:
    adam_history4 = pickle.load(f)

# %%
plt.plot(sgd_history4['loss'], label='SGD')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-sgd.png')

plt.clf()

plt.plot(batch_history4['loss'], label='Batch')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-batch.png')

plt.clf()

plt.plot(sgd_moment_history4['loss'], label='SGD with momentum')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-sgd-moment.png')

plt.clf()

plt.plot(nag_history4['loss'], label='NAG')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-nag.png')

plt.clf()

plt.plot(adagrad_history4['loss'], label='AdaGrad')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-adagrad.png')

plt.clf()

plt.plot(rmsprop_history4['loss'], label='RMSProp')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-rmsprop.png')

plt.clf()

plt.plot(adam_history4['loss'], label='Adam')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Arch4-adam.png')

plt.clf()


# %%
plt.plot(sgd_history4['loss'], label='SGD')
plt.plot(batch_history4['loss'], label='Batch')
plt.plot(sgd_moment_history4['loss'], label='SGD with momentum')
plt.plot(nag_history4['loss'], label='NAG')
plt.plot(adagrad_history4['loss'], label='AdaGrad')
plt.plot(rmsprop_history4['loss'], label='RMSProp')
plt.plot(adam_history4['loss'], label='Adam')

plt.title('Architecture 4')
# plt.xlim((0,15))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Arch4-super.png")

# %%


# %%


# %%
test_path='./Group_10/test'
train_path='./Group_10/train'
val_path='./Group_10/val'
with open('test_data', mode='rb') as f:
    test_data = pickle.load(f)
with open('train_data', mode='rb') as f:
    train_data = pickle.load(f)
with open('val_data', mode='rb') as f:
    val_data = pickle.load(f)

with open('test_labels', mode='rb') as f:
    test_labels = pickle.load(f)
with open('train_labels', mode='rb') as f:
    train_labels = pickle.load(f)
with open('val_labels', mode='rb') as f:
    val_labels = pickle.load(f)

print('Summary of data')
print(f'No. of train images: {len(train_data)}')
print(f'No. of test images: {len(test_data)}')
print(f'No. of val images: {len(val_data)}')


# %%
model = keras.models.load_model("SGD_model2.h5")
model.evaluate(train_data, train_labels), model.evaluate(val_data, val_labels)


# %%



