# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:10:56 2017

@author: chetu
"""

import train
import test

# Train OCR
train_images = ["a.bmp", "d.bmp", "f.bmp", "h.bmp", "k.bmp", "m.bmp",
                "n.bmp", "o.bmp", "p.bmp", "q.bmp", "r.bmp", "s.bmp",
                "u.bmp", "w.bmp", "x.bmp", "z.bmp"];
                
train_features, train_labels, train_mean, train_std = train.train(train_images)

# Test Recognition and compare to ground truth
bbox_list, labels = test.test("test1.bmp", train_features, train_labels, train_mean, train_std)
test.evaluate('test1_gt.pkl', bbox_list, labels)

bbox_list, labels = test.test("test2.bmp", train_features, train_labels, train_mean, train_std)
test.evaluate('test2_gt.pkl', bbox_list, labels)