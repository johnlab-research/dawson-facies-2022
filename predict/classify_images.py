# Adapted from pyimagesearch.com
# http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

# Import libraries
import os, sys
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from tf.keras.models import load_model
from tf.keras.applications import imagenet_utils
from tf.keras.applications.inception_v3 import preprocess_input
from tf.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help = "name of model to use")
args = vars(ap.parse_args())

# Define a dictionary that maps model names to their classes inside Keras
MODELS = {
	"vgg19": "carbonates_vgg19_model_ft.h5",
	"inceptionv3": "carbonates_inception_model_ft.h5",
	}

# Check for valid image
print("[INFO] checking image - {}...".format(args["image"]))
if os.path.exists(args["image"]) ==  False:
	print("Image does not exist")
	sys.exit()

# Ensure a valid model name was supplied via command line argument
print("[INFO] checking model file -  {}...".format(args["model"]))
if args["model"] not in MODELS.keys():
	print("The --model command line argument should "
		"be a key in the 'MODELS' dictionary. "
		"Choices are:\n" + "vgg19\n" + "inceptionv3")
	sys.exit()
model_file = "carbonates_" + args["model"] + "_model_ft.h5"
if os.path.exists(model_file) == False:
	print(args["model"] + " model file is missing")
	sys.exit()

# Initialise the image input shape (224x224 pixels) along with the pre-processing function (this may need to be changed based on which model is used to classify the image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# If using the InceptionV3 network, then the input shape needs to be set to (299x299) [rather than (224x224)] and use a different image processing function
if args["model"] in ("inception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

# Load the model from disk 
# (NOTE: if this is the first time this script is used for a given model, the weights will need to be downloaded first -- depending on which network is being used, 
# the weights can be 96-574MB, so be patient.)
print("[INFO] loading {}...".format(args["model"]))
model_file = MODELS[args["model"]]
loaded_model = load_model(model_file)

# Load the input image using the Keras helper utility while ensuring the image is resized to `inputShape` (the required input dimensions for the ImageNet pre-trained network)
print("[INFO] loading and pre-processing image...")

img = load_img(args["image"])
if os.path.exists(args["image"]):
	image = load_img(args["image"], target_size=inputShape)
else:
	print("Image does not exist")
	sys.exit()
image = img_to_array(image)

# Expand the dimension by making the shape (1, inputShape[0], inputShape[1], 3) so it can be passed through thenetwork
image = np.expand_dims(image, axis=0)

# Preprocess the image using the appropriate function based on the model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)

# Classify the image
print("[INFO] classifying image with {}...".format(args["model"]))
#preds = loaded_model.predict(image)
print("[INFO] close Image to display result...")
preds = [np.argmax(loaded_model.predict(image))]
print(preds)

# Display the image
plt.imshow(img)
plt.axis('off')
plt.show()

# Display the classification results
for preds in preds:
	if preds == 0:
		print("[RESULT] Image predicted to be boundstone")
	elif preds == 1:
		print("[RESULT] Image predicted to be floatstone")
	elif preds == 2:
		print("[RESULT] Image predicted to be grainstone")
	elif preds == 3:
		print("[RESULT] Image predicted to be mudstone")
	elif preds == 4:
		print("[RESULT] Image predicted to be packstone")
	elif preds == 5:
		print("[RESULT] Image predicted to be rudstone")
	elif preds == 6:
		print("[RESULT] Image predicted to be wackestone")
	else:
		print("[RESULT] Image does not fall into any of these seven Dunham classes:\n"
		+ "boundstone\n" + "floatstone\n" + "grainstone\n"
		+ "mudstone\n" + "packstone\n" + "rudstone\n" + "wackestone")