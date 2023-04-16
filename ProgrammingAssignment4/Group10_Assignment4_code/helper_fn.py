import numpy as np
import cv2
import os

class my_pca:
    '''
    Procedure for PCA. To reduce an (n x d) matrix X to k dimensions (where k<d), we perform the following steps:

    1) Center X.
    2) Use the SVD to find X = U@S@Vt.
    3) Compute the principal components U@S.
    4) Keep the first k columns of U@S.
    '''
    def __init__(self, data_matrix):
        # self.data_matrix = data_matrix
        self.mean_vec = np.mean(data_matrix, axis=0) # 784 dimentional vector
        data_matrix = data_matrix - self.mean_vec
        U, sigma, Vt = np.linalg.svd(data_matrix, full_matrices=False)
        self.principal_components = U @ np.diag(sigma)
        return None
    
    def components(self, k):
        return self.principal_components[:k, :] # k-columns are the k components
    def transform(self, data, k):
        '''
        Arguments:
        Data to augment
        k-components to use
        
        Returns:
        Reduced representation
        '''
        centered_data = data - self.mean_vec
        return self.components(k) @ centered_data #transformation
    

def data_import(data_path):
    class_labels = os.listdir(data_path) # reads directory names as class-labels
    data=[]
    labels=[]
    for class_ in class_labels:
        if class_ == '.DS_Store':
            continue
        class_path = data_path+'/'+class_
        imgs = os.listdir(class_path) # reads images names to read
        for img in imgs:
            if img == '.DS_Store':
                continue
            data.append(cv2.imread(class_path+'/'+img, cv2.IMREAD_GRAYSCALE))
            labels.append(int(class_))

    return np.array(data), np.array(labels)

def relabel(labels):
    label_map = { # old : new
        0 : 0,
        1 : 1,
        2 : 2,
        6 : 3,
        7 : 4,
    }
    for i, l in enumerate(labels):
        labels[i] = label_map[l]
    return labels