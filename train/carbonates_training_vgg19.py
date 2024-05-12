# Setup

# Import libraries

import os
import sys
import glob as glob
import numpy as np
import datetime
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tf.keras.applications.vgg19 import VGG19, preprocess_input
from tf.keras.models import Model
from tf.keras.layers import Dense, GlobalAveragePooling2D
from tf.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tf.keras.optimizers import SGD

from scipy.interpolate import spline
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

tensorflow.__version__

tf.keras.__version__

#----------------------------------------------------------------------------

# Load and process dataset

import pathlib

dataset_path = '/content/drive/MyDrive/carbonate_cores'
data_root = pathlib.Path(dataset_path)

batch_size = 32
img_height = 224
img_width = 224


# Load data into the model

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = np.array(train_ds.class_names)
print(class_names)


# Use rescaling preprocessing layer to achieve float inputs in the [0, 1] range

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # where x—images, y—labels.

# Finish the input pipeline by using buffered prefetching to yield the data from disk without I/O blocking issues

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#----------------------------------------------------------------------------

# Model setup

# Setup transfer learning on pre-trained ImageNet VGG19 model 
# Remove fully connected layer and replace with softmax for classifying 7 classes

num_classes = len(class_names)

vgg19_model = VGG19(weights = 'imagenet', include_top = False)
x = vgg19_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)
model = Model(input = vgg19_model.input, output = predictions)

# Freeze all layers of the pre-trained model
for layer in vgg19_model.layers:
    layer.trainable = False
    
model.summary()

predictions = model(image_batch)

predictions.shape    

#----------------------------------------------------------------------------

# Model training 

# Compile the model using Adam optimizer
model.compile(
  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

# Train the model
NUM_EPOCHS = 15

now = datetime.datetime.now
t = now()
tl_history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    class_weight='auto')
print('Training time: %s' % (now() - t))

#----------------------------------------------------------------------------

# Evaluate performance

predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
print(predicted_label_batch)

# Plot model predictions

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")

model.save('carbonates_vgg19_model_tl.h5')

xfer_acc = tl_history.history['acc']
val_acc = tl_history.history['val_acc']
xfer_loss = tl_history.history['loss']
val_loss = tl_history.history['val_loss']
epochs = range(len(xfer_acc))

x = np.array(epochs)
y = np.array(xfer_acc)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_acc)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Transfer Learning - Training and Validation Accuracy')
plt.legend(loc = 'lower left', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0,1.02)

plt.figure()
x = np.array(epochs)
y = np.array(xfer_loss)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_loss)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Transfer Learning - Training and Validation Loss')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,max(y))
plt.show()

#----------------------------------------------------------------------------

# Fine tuning

for i, layer in enumerate(vgg19_model.layers):
   print(i, layer.name)
   
# Freeze lower layers of pre-trained model
for layer in model.layers[:12]:
    layer.trainable = False
for layer in model.layers[12:]:
    layer.trainable = True
    
# Compile revised model using Adam optimizer with a learing rate of 0.000001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.000001)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

# Train the revised model
now = datetime.datetime.now
t = now()
ft_history = model.fit_generator(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    class_weight='auto')
print('Training time: %s' % (now() - t))

#----------------------------------------------------------------------------

# Evaluate performance of fine tuned model

predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
print(predicted_label_batch)

# Plot model predictions

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")

model.save('carbonates_inception_v3_model_ft.h5')

ft_acc = ft_history.history['acc']
val_acc = ft_history.history['val_acc']
ft_loss = ft_history.history['loss']
val_loss = ft_history.history['val_loss']
epochs = range(len(ft_acc))

x = np.array(epochs)
y = np.array(xfer_acc)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_acc)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Fine-Tuning - Training and Validation Accuracy')
plt.legend(loc = 'lower left', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0,1.02)

plt.figure()
x = np.array(epochs)
y = np.array(xfer_loss)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_loss)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Fine-Tuning - Training and Validation Loss')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,max(y))
plt.show()