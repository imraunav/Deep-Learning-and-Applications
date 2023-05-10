import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import process_mfcc, data_import_cv


def main():
    data_path = "./ProgrammingAssignment6/CS671-DLA-Assignment4-Data-2022/CV_Data"
    x_train, y_train, x_test, y_test = data_import_cv(data_path)
    # print(len(y_train))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(x_train[i])
    # plt.show()

    

    return None
if __name__ == "__main__":
    main()