# Transfer Learning for Geological Images

## Acknowledgement

## Description
The code in this repository can be used to easily perform transfer learning using any of the built-in TensorFlow-Keras image classification models used in the Dawson *et al*., 2023 paper. 

## Built-In Models
The built-in models made available here:

| Model  | Size  |  Top-1 Accuracy  |  Top-5 Accuracy  |  Parameters  |  Depth  |
| -------------     | ------------- | -------------| ------------- | ------------- | ------------- |
| VGG16    | 528 MB    |  0.713    | 0.901    | 138.4M    | 16    |
| VGG19    | 549 MB    |  0.727    | 0.910    | 143.7M    | 19    |
| ResNet50    |  98 MB    | 0.749    | 0.921    | 25.6M   |  107    |
| ResNet101    |  171 MB    | 0.764   | 0.928    | 44.7M     | 209    |
| ResNet152    | 232 MB    | 0.766   | 0.931    | 60.4M    |  311    |
| InceptionV3    | 92 MB    | 0.779    | 0.937    | 23.9M    |  159    |
| DenseNet121    | 33 MB    | 0.750    | 0.923    | 8.1M    | 121    |
| DenseNet169    | 57 MB    | 0.762    | 0.932    | 14.3M    |  169    |
| DenseNet201    | 80 MB    | 0.773    | 0.936    | 20.2M    |  201    |


## Files and Directories


- **comined-model.py:** Training and prediction mode

- **utils.py:** Helper utility functions


## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

- Keras `sudo pip install tf.keras` 

## Usage
To use this code, please ensure you have set up your dataset in folders following this structure:

    ├── "dataset_name"                   
    |   ├── train
    |   |   ├── class_1_images
    |   |   ├── class_2_images
    |   |   ├── class_X_images
    |   |   ├── .....
    |   ├── val
    |   |   ├── class_1_images
    |   |   ├── class_2_images
    |   |   ├── class_X_images
    |   |   ├── .....
    |   ├── test
    |   |   ├── class_1_images
    |   |   ├── class_2_images
    |   |   ├── class_X_images
    |   |   ├── .....

Then you can simply run `combined-model.py`. 

Check out the optional command line arguments:

```
usage: main.py [-h] [--num_epochs NUM_EPOCHS] [--mode MODE] [--image IMAGE]
               [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
               [--resize_height RESIZE_HEIGHT] [--resize_width RESIZE_WIDTH]
               [--batch_size BATCH_SIZE] [--dropout DROPOUT] [--h_flip H_FLIP]
               [--v_flip V_FLIP] [--rotation ROTATION] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for, default=25.
  --mode MODE           Select "train" or "predict" mode, default="train". Note that for
                        prediction mode you have to specify an image to run
                        the model on.
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Option to continue training from a checkpoint, default=False.
  --dataset DATASET     Dataset you are using, default="carbonate_cores".
  --resize_height RESIZE_HEIGHT
                        Height of cropped input image to network, default=299.
  --resize_width RESIZE_WIDTH
                        Width of cropped input image to network, default=299. 
  --batch_size BATCH_SIZE
                        Number of images in each batch, default=32.
  --dropout DROPOUT     Dropout ratio, default=0.2.
  --h_flip H_FLIP       Option to randomly flip the image horizontally for
                        data augmentation, default=False.
  --v_flip V_FLIP       Option to randomly flip the image vertically for data
                        augmentation, default=False.
  --rotation ROTATION   Option to randomly rotate the image for data
                        augmentation, default=0.0.
  --model MODEL         Your pre-trained classification model of choice, default=InceptionV3.

```

## TensorFlor + Keras 2 backwards compatibility

From TensorFlow 2.0 to TensorFlow 2.15 (included), doing `pip install tensorflow` will also install the corresponding version of Keras 2 – for instance, `pip install tensorflow==2.14.0` will install `keras==2.14.0`. That version of Keras is then available via both import keras and from tensorflow import keras (the `tf.keras` namespace).

Starting with TensorFlow 2.16, doing `pip install tensorflow` will install Keras 3. When you have TensorFlow >= 2.16 and Keras 3, then by default `from tensorflow import keras` (`tf.keras`) will be Keras 3.

Meanwhile, the legacy Keras 2 package is still being released regularly and is available on PyPI as `tf_keras` (or equivalently `tf-keras` – note that `-` and `_` are equivalent in PyPI package names). To use it, you can install it via `pip install tf_keras` then import it via `import tf_keras as keras`.

## References

* VGG16 and VGG19 - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)
* InceptionV3 - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) (CVPR 2016)
* ResNet - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
* DenseNet - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

**Note: each TF-Keras Application expects a specific kind of input preprocessing.**

For VGG16, call `tf.keras.applications.vgg16.preprocess_input` on your inputs before passing them to the model. `vgg16.preprocess_input` will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.

For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your inputs before passing them to the model. `vgg19.preprocess_input` will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.

For InceptionV3, call `tf.keras.applications.inception_v3.preprocess_input` on your inputs before passing them to the model. `inception_v3.preprocess_input` will scale input pixels between -1 and 1.

For ResNet, call `tf.keras.applications.resnet.preprocess_input` on your inputs before passing them to the model. `resnet.preprocess_input` will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.

For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your inputs before passing them to the model.