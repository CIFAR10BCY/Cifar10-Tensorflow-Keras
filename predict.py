# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:02:01 2019

@author: ck
"""

import os
import numpy as np 
import tensorflow as tf

import datetime, os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,  Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import datasets, layers, models

from tensorflow.keras import regularizers, optimizers

from IPython.display import display
from tensorflow.keras.preprocessing.image import array_to_img 
#Tensorboard
from tensorflow.keras.callbacks import TensorBoard
from time import strftime


import matplotlib.pyplot as plt




LOG_DIR = 'tensorboard_cifar_logs\\'


label_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog',
               'Horse', 'Ship', 'Truck']


(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()


x_train_all, x_test = x_train_all / 255.0 , x_test /255.0 


x_val = x_train_all[:10000] #Validation size
y_val = y_train_all[:10000]

x_train = x_train_all[10000:]
y_train = y_train_all[10000:]

x_train_xs = x_train[:1000] #SMALL TRAIN SIZE
y_train_xs = y_train[:1000]

load_model = tf.keras.models.load_model('saved_model/my_model')


load_model = tf.keras.models.load_model('saved_model/my_model')
loss, acc = load_model.evaluate(x_val, y_val, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

test = np.expand_dims(x_val[0], axis=0)
test.shape
np.set_printoptions(precision=3)


load_model.predict(test)


for number in range(10):
    test_img = np.expand_dims(x_val[number], axis=0)
    predicted_val = load_model.predict_classes(test_img)[0]
    print(f"Actual value: {y_val[number][0]} vs. predicted: {predicted_val}")
    
for number in range(10):
    test_img = np.expand_dims(x_val[number], axis=0)
    predicted_val = load_model.predict_classes(test_img)[0]
    print(f"Actual value: {label_names[y_val[number][0]]} vs. predicted: {label_names[predicted_val]}")
    
load_model.metrics_names

test_loss, test_acc = load_model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss : {test_loss:0.3}, Test Accuracy: {test_acc:0.2%}")


from sklearn.metrics import confusion_matrix

predictions = load_model.predict_classes(x_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions )

import itertools


plt.figure(figsize=(7,7))
plt.imshow(conf_matrix, cmap=plt.cm.Reds)
plt.title("Confusion Matrix", fontsize=18)
plt.ylabel("Actual Labels", fontsize=14)
plt.xlabel("Predicted labels", fontsize=14)
plt.yticks(np.arange(10), label_names)
plt.xticks(np.arange(10), label_names)

plt.colorbar()
for i, j in itertools.product(range(10), range(10)):
    plt.text(j, i, conf_matrix[i,j], horizontalalignment='center', color="white" if conf_matrix[i,j]> 350 else 'black')

plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Doğruluk (accuracy)  :{:.2f}".format(accuracy_score(y_test, predictions)))
print("Keskinlik (precision) :{:.2f}".format(precision_score(y_test, predictions, average='weighted')))
print("Duyarlılık (recall) :{:.2f}".format(recall_score(y_test, predictions, average='weighted')))
print("f_1 skoru (f1) :{:.2f}".format(f1_score(y_test, predictions, average='weighted')))


from PIL import Image

def im_read(name):
    image = Image.open(name)
    plt.imshow(image)
    plt.show()
    image = Image.fromarray(np.uint8(image))
    image = image.resize((32, 32))

    resize_frame = np.asarray(image)
    print(resize_frame.shape)
    plt.imshow(resize_frame)
    plt.show()
    return resize_frame

def convertCIFAR10Data(image):
    img = image.astype('float32')
    img /= 255
    c = np.zeros(32*32*3).reshape((1,32,32,3))
    print(c.shape)
    c[0] = img
    print(c[0].shape)
    return c

image1 = im_read("cat.jpg")
data1 = convertCIFAR10Data(image1)

p1 = load_model.predict_classes(data1)
print(f"Resim {label_names[p1[0]]}, olarak tahmin edildi.")


image2 = im_read("ship.jpg")
data2 = convertCIFAR10Data(image2)
p2 = load_model.predict_classes(data2)
print(f"Resim {label_names[p2[0]]}, olarak tahmin edildi.")




