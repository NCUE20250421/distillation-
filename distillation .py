import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import shutil
import tensorflow as tf

# Use keras directly instead of tensorflow.python.keras
from keras.models import Model
from keras.layers import Conv2D, Dropout, MaxPool2D, Input, Conv2DTranspose, Concatenate
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers

from sklearn.model_selection import train_test_split
import random
import h5py
from IPython.display import display
from PIL import Image as im
import datetime

train_path='C:/Users/AI_Server07/Desktop/distillation-/data/train'

for i, folder in enumerate(os.listdir(train_path)):
    for j, img in enumerate(os.listdir(train_path+"/"+folder)):
        filename = train_path+"/"+folder + "/" + img
        img= im.open(filename)
        ax = plt.subplot(3,4,4*i+j+1)
        ax.set_xlabel(folder+ ' '+ str(img.size[0]) +'x'+ str(img.size[1]))
        plt.imshow(img, 'gray')
        ax.set_xlabel(folder+ ' '+ str(img.size[0]) +'x'+ str(img.size[1]))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        #plt.axis('off')
        img.close()
        if j>2:
            break