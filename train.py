# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:25:29 2017

@author: chetu
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utilities

inv_label_map = {0:'a', 1:'d', 2:'f', 3:'h', 4:'k',
             5:'m', 6:'n', 7:'o', 8:'p', 9:'q',
             10:'r', 11:'s', 12:'U', 13:'w', 14:'x', 15:'z'}
             
def train(train_images):
    train_features = np.array([[0]*utilities.num_features]);
    train_labels = [];

    for lab, images in enumerate(train_images):
        print("training image : ", lab)
        features, labels = extract_features("H1-16images/" + images,lab)
        train_features = np.append(train_features,features,axis=0)
        train_labels.extend(labels)
    
    train_features = np.delete(train_features, 0, 0)
    
    train_mean = np.mean(train_features, axis = 0)
    train_std = np.std(train_features, axis = 0)
    
    train_features = (train_features - train_mean)/train_std
    
    #compute distances between training images
    D = cdist(train_features, train_features)
    io.imshow(D)
    plt.title('Distance Matrix')
    io.show()
    
    D_index = np.argsort(D, axis=1)
    pred_labels_train = utilities.classify(D_index, train_labels)
    
    #evaluate on training images    
    succ = 0
    for i in range(0,len(train_labels)):
        if(train_labels[i] == pred_labels_train[i]):
            succ = succ+1
    print("training recognition rate: ", succ/len(train_labels), len(train_labels))
    
    # Confusion matrix
    confM = confusion_matrix(train_labels, pred_labels_train)
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()
    
#    Display labeled training images
#    curidx = 0
#    for lab, images in enumerate(train_images):
#        curidx = label_train("H1-16images/" + images, pred_labels_train, curidx)
        
    return train_features, train_labels, train_mean, train_std;

def label_train(image_path, pred_labels_train, curidx):
    img_binary = utilities.get_threshold(image_path)
    io.imshow(img_binary)
    
    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
    ax = plt.gca()
    
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        if (maxc - minc) < 10 or (maxr - minr) < 10:
            continue
        center_x = (minr + maxr)/2
        center_y = maxc + 5
        ax.text(center_y, center_x, str(inv_label_map[pred_labels_train[curidx]]), fontsize=8, color='yellow')
        curidx = curidx + 1
        
    io.show()  
    return curidx
    
def extract_features(image_path, class_label):
    features = np.array([[0]*utilities.num_features])
    labels = []
    
    img_binary = utilities.get_threshold(image_path)
    
    img_label = label(img_binary, background=0)

    regions = regionprops(img_label)
    
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        
        if (maxc - minc) < 10 or (maxr - minr) < 10:
            continue
        
        roi = img_binary[minr:maxr, minc:maxc]
        curr_features = utilities.extract_features(roi, props)
        
        features = np.append(features,curr_features,axis=0)
        labels.append(class_label)

    features = np.delete(features, 0, 0)
    return features, labels;