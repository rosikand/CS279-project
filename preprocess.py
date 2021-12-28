"""
File: preprocess.py
------------------
This file contains several useful functions that are used
for preprocessing the raw data. 
"""

import pandas as pd
import pickle
from PIL import Image
import os
import numpy as np
import cv2

# this function will return a 28-vector where there is a 1 at the index of each label 
def get_label(path_string, df_list):
    label_vector = []
    for i in range(28):
        label_vector.append(0)
        
    labels = []
    for df_elem in df_list:
        df_elem_string = df_elem[0]
        if df_elem_string == path_string:
            labels = df_elem[1:] 
    if len(labels) == 1:
        labels = labels[0].split(" ")
    labels = [int(item) for item in labels]
    
    # one-hot encode 
    for num_val in labels:
        label_vector[num_val] = 1
    
    return label_vector
    

def preprocess_data(data_list, labels_list):  
    """
    This block of code was inspired by code from here: 
    https://www.kaggle.com/allunia/protein-atlas-exploration-and-baseline
    """
    R = np.array(Image.open(data_list[3]))
    G = np.array(Image.open(data_list[2]))
    B = np.array(Image.open(data_list[1]))
    Y = np.array(Image.open(data_list[0]))

    images = np.zeros(shape=(512,512,4))
    images[:,:,0] = G
    images[:,:,1] = R
    images[:,:,2] = B
    images[:,:,3] = Y
    images = np.divide(images, 255)  # normalize 

    path_string = data_list[2]
    path_string = path_string[13:path_string.rfind('_')]
    
    # get label 
    label = get_label(path_string, labels_list)
    return images, label


def main():
    dirs = os.listdir('data_subset')
    data_list = []
    for elem in dirs:
        if elem.endswith('.png'):
            data_list.append("data_subset/" + elem)
    data_list.sort()
    
    master_list = []
    for i in range(len(data_list)):
        if (i % 4 == 0) and (i != 0):
            i = i - 1
            curr_image = data_list[i-4:i] 
            master_list.append(curr_image)
    master_list = master_list[1:]
    
    
    df = pd.read_csv('train.csv')
    df_list = df.values.tolist()

    processed_dataset = []
    for image_collection in master_list:
        curr_sample = preprocess_data(image_collection, df_list)  # tuple with X, y
        processed_dataset.append(curr_sample)
    
    out_file = open("binary_files/subset.pkl", "wb")
    pickle.dump(processed_dataset, out_file)
    out_file.close()

              
        
if __name__ == '__main__':
  main()