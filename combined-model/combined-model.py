from __future__ import print_function

#----------------------------------------------------------------------------

# Set up

# Import networks
from tf.keras.preprocessing import image
from tf.keras.applications.densenet121 import DenseNet121
from tf.keras.applications.densenet169 import DenseNet169
from tf.keras.applications.densenet201 import DenseNet201
from tf.keras.applications.inception_v3 import InceptionV3
from tf.keras.applications.resnet50 import ResNet50
from tf.keras.applications.resnet101 import ResNet101
from tf.keras.applications.resnet152 import ResNet152
from tf.keras.applications.vgg16 import VGG16
from tf.keras.applications.vgg19 import VGG19
from tf.keras.preprocessing.image import ImageDataGenerator

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

# Files
import utils

# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train for, default=25.')
parser.add_argument('--mode', type=str, default="train", help='Select "train" or "predict" mode, default="train". \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Option to continue training from a checkpoint, default=False.')
parser.add_argument('--dataset', type=str, default="Pets", help='Dataset you are using, default="carbonate_cores".')
parser.add_argument('--resize_height', type=int, default=299, help='Height of input image to network, default=299.')
parser.add_argument('--resize_width', type=int, default=299, help='Width of input image to network, default=299.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch, default=32.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio, default=0.2.')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Option to randomly flip the image horizontally for data augmentation, default=False.')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Option to randomly flip the image vertically for data augmentation, default=False.')
parser.add_argument('--rotation', type=float, default=0.0, help='Option to randomly rotate the image for data augmentation, default=0.0.')
parser.add_argument('--model', type=str, default="InceptionV3", help='Your pre-trained classification model of choice, default=InceptionV3.')
args = parser.parse_args()


# Global settings
BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
TRAIN_DIR = args.dataset + "/train/"
VAL_DIR = args.dataset + "/val/"

preprocessing_function = None
base_model = None

#----------------------------------------------------------------------------

# Model selection
if args.model == "VGG16":
    from tf.keras.applications.vgg16 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "VGG19":
    from tf.keras.applications.vgg19 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet50":
    from tf.keras.applications.resnet50 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet101":
    from tf.keras.applications.resnet101 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet152":
    from tf.keras.applications.resnet152 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))   
elif args.model == "InceptionV3":
    from tf.keras.applications.inception_v3 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet121":
    from tf.keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet169":
    from tf.keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet201":
    from tf.keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
else:
    ValueError("The model you requested was not made available in this study.")

#----------------------------------------------------------------------------

# Prepare the model 

# Train model   
if args.mode == "train":
    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Resize Height -->", args.resize_height)
    print("Resize Width -->", args.resize_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tRotation -->", args.rotation)
    print("")

    # Create directories, if needed
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")

    # Prepare image data generators
    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocessing_function,
      rotation_range=args.rotation,
      horizontal_flip=args.h_flip,
      vertical_flip=args.v_flip
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)


    # Save the list of classes for prediction mode later
    class_list = utils.get_subfolders(TRAIN_DIR)
    utils.save_class_list(class_list, model_name=args.model, dataset_name=args.dataset)

    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))

    if args.continue_training:
        finetune_model.load_weights("./checkpoints/" + args.model + "_model_weights.h5")

    adam = Adam(lr=0.0001)
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    num_train_images = utils.get_num_files(TRAIN_DIR)
    num_val_images = utils.get_num_files(VAL_DIR)

    def lr_decay(epoch):
        if epoch%20 == 0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr/2)
            print("LR changed to {}".format(lr/2))
        return K.get_value(model.optimizer.lr)

    learning_rate_schedule = LearningRateScheduler(lr_decay)

    filepath="./checkpoints/" + args.model + "_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    callbacks_list = [checkpoint]


    history = finetune_model.fit_generator(train_generator, epochs=args.num_epochs, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE, 
        validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)


    plot_training(history)

# Prediction mode
elif args.mode == "predict":

    if args.image is None:
        ValueError("You must pass an image path when using prediction mode.")

    # Create directories, if needed
    if not os.path.isdir("%s"%("Predictions")):
        os.makedirs("%s"%("Predictions"))

    # Read in the image
    image = cv2.imread(args.image,-1)
    save_image = image
    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))

    class_list_file = "./checkpoints/" + args.model + "_" + args.dataset + "_class_list.txt"

    class_list = utils.load_class_list(class_list_file)
    
    finetune_model = utils.build_finetune_model(base_model, len(class_list))
    finetune_model.load_weights("./checkpoints/" + args.model + "_model_weights.h5")

    # Run the classifier and print results
    st = time.time()

    out = finetune_model.predict(image)

    confidence = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]

    run_time = time.time()-st

    print("Predicted class = ", class_name)
    print("Confidence = ", confidence)
    print("Run time = ", run_time)
    cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)