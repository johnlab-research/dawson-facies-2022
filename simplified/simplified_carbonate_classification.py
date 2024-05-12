# Simplified transfer learning using TensorFlow Hub 

# TensorFlow Hub is a repository of pre-trained TensorFlow models.
# Here, we use InceptionV3. All available models can be found at https://www.kaggle.com/models?tfhub-redirect=true

#----------------------------------------------------------------------------

# Setup

# Import libraries

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import time

import PIL.Image as Image
import tensorflow_hub as hub

import datetime

%load_ext tensorboard

# Import model

inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

classifier_model = inception_v3

IMAGE_SHAPE = (299 , 299)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])


#----------------------------------------------------------------------------

# Load and process dataset

import pathlib

dataset_path = '/content/drive/MyDrive/carbonate_cores'
data_root = pathlib.Path(dataset_path)

batch_size = 32
img_height = 299
img_width = 299


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

# Download headless model

# TensorFlowHub also distributes models without the top classification layer. 
# These can be used to easily perform transfer learning.

# Select an InceptionV3 pre-trained model from TensorFlow Hub
# Any compatible image feature vector model from TensorFlow Hub (https://www.kaggle.com/models?task=16701&query=tf2&tfhub-redirect=true) will work here.

inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_model = inception_v3

# Create the feature extractor

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(299, 299, 3),
    trainable=False)

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

#----------------------------------------------------------------------------

# Attach classification head

num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

predictions = model(image_batch)

predictions.shape

#----------------------------------------------------------------------------

# Model training

# Configure the training process

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

# Train the model

NUM_EPOCHS = 15

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)

# Start the TensorBoard to view how metrics change with each epoch

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/fit

#----------------------------------------------------------------------------

# Check predictions

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

#----------------------------------------------------------------------------

# Export and reload model

t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path)

export_path

reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()

reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
print(reloaded_predicted_label_batch)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(reloaded_predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")