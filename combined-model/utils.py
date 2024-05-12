# Setup

# Import layers
from tf.keras.layers import Dense
from tf.keras.layers import Activation 
from tf.keras.layers import Flatten
from tf.keras.layers import Dropout
from tf.keras import backend as K

# Import other 
from tf.keras import optimizers
from tf.keras import losses
from tf.keras.optimizers import SGD
from tf.keras.optimizers import Adam
from tf.keras.models import Sequential
from tf.keras.models import Model
from tf.keras.models import load_model
from tf.keras.callbacks import ModelCheckpoint
from tf.keras.callbacks import LearningRateScheduler

# Import utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import glob
import os
import sys
import csv
import cv2
import time
import datetime

#----------------------------------------------------------------------------

# Define functions

def save_class_list(class_list, model_name, dataset_name):
    class_list.sort()
    target=open("./checkpoints/" + model_name + "_" + dataset_name + "_class_list.txt",'w')
    for c in class_list:
        target.write(c)
        target.write("\n")

def load_class_list(class_list_file):
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

# Get a list of subfolders in the directory
def get_subfolders(directory):
    subfolders = os.listdir(directory)
    subfolders.sort()
    return subfolders

# Get number of files by searching directory recursively
def get_num_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

# Add on new FC layers with dropout for fine tuning
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')