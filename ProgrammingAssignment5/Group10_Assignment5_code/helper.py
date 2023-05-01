import cv2
import numpy as np
import os 

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
            img_mat = cv2.imread(class_path+'/'+img)
            data.append(cv2.resize(img_mat, (224, 224)))
            labels.append(class_)

    return np.array(data), labels

def relabel(labels):
    label_map = {
        # old : new
        "car_side" : 0,
        "ewer" : 1,
        "grand_piano" : 2,
        "helicopter" : 3,
        "laptop" : 4,
    }
    # label_map ={
    #     "bonsai" : 0,
    #     "buddha" : 1,
    #     "car_side" : 2,
    #     "scorpion" : 3,
    #     "starfish" : 4,
    # }
    for i, l in enumerate(labels):
        labels[i] = label_map[l]
    return np.array(labels)