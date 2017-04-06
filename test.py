# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:46:32 2017

@author: chetu
"""

import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io
import matplotlib.pyplot as plt
import pickle
import utilities
from matplotlib.patches import Rectangle

label_map = {'a':0, 'd':1, 'f':2, 'h':3, 'k':4,
             'm':5, 'n':6, 'o':7, 'p':8, 'q':9,
             'r':10, 's':11, 'U':12, 'w':13, 'x':14, 'z':15}

inv_label_map = {0:'a', 1:'d', 2:'f', 3:'h', 4:'k',
             5:'m', 6:'n', 7:'o', 8:'p', 9:'q',
             10:'r', 11:'s', 12:'U', 13:'w', 14:'x', 15:'z'}
             
def test(image, train_features, train_labels, train_mean, train_std):
    features, bbox_list = extract_features("H1-16images/" + image)
    features = (features - train_mean)/train_std
    D = cdist(features, train_features)
    D_index = np.argsort(D, axis=1)
    pred_labels_test = utilities.classify(D_index, train_labels)
    ax = plt.gca()
    
    for idx,box in enumerate(bbox_list):
        center_x = (box[0] + box[2])/2
        center_y = box[3] + 10
        ax.text(center_y, center_x, str(inv_label_map[pred_labels_test[idx]]), fontsize=8, color='yellow')
    io.show()

    return bbox_list, pred_labels_test
    
def extract_features(image_path):
    features = np.array([[0]*utilities.num_features])
    bbox_list = np.array([[0,0,0,0]])
    
    img_binary = utilities.get_threshold(image_path)
    
    io.imshow(img_binary)
    img_label = label(img_binary, background=0)
    
#    print ('connected components : ', np.amax(img_label))
    
    regions = regionprops(img_label)
    ax = plt.gca()
    
    for props in regions:
        minr, minc, maxr, maxc = props.bbox

        if (maxc - minc) < 10 or (maxr - minr) < 10:
            continue
        
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        
        roi = img_binary[minr:maxr, minc:maxc]
        curr_features = utilities.extract_features(roi, props)
        features = np.append(features,curr_features,axis=0)
        bbox_list = np.append(bbox_list, np.array([[minr, minc, maxr, maxc]]),axis=0)
    
    print ('components : ', len(features)-1)
    
    features = np.delete(features, 0, 0)
    bbox_list = np.delete(bbox_list, 0, 0)
    
    return features,bbox_list;

def evaluate(gtfile, bbox_list, labels):
    with open(gtfile, 'rb') as pkl_file:
        mydict = pickle.load(pkl_file,encoding='latin1') 
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']
    
    succ = 0
    for idx,center in enumerate(locations):
        for l,box in enumerate(bbox_list):
            if (center[1] > box[0]) & (center[1] < box[2]) & (center[0] > box[1]) & (center[0] < box[3]) & (label_map[classes[idx]] == labels[l]):
                succ = succ + 1
    print ("recognition rate: ", succ/len(classes))
    print ("####################################")
    