import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

from helper_fn import data_import

test_path='./ProgrammingAssignment4/Group_10/test'
train_path='./ProgrammingAssignment4/Group_10/train'
val_path='./ProgrammingAssignment4/Group_10/val'

test_data, test_labels = data_import(test_path)
train_data, train_labels = data_import(train_path)
val_data, val_labels = data_import(val_path)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Val data shape: {val_data.shape}")

