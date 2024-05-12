# Transfer Learning for Geological Images

## Description
The code in this repository can be used to easily perform transfer learning using any of the built-in TensorFlow-Keras image classification models used in the Dawson et al., 2023 paper.

## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

- Keras `sudo pip install tf.keras` 

### NB: TensorFlor + Keras 2 backwards compatibility

From TensorFlow 2.0 to TensorFlow 2.15 (included), doing `pip install tensorflow` will also install the corresponding version of Keras 2 – for instance, `pip install tensorflow==2.14.0` will install `keras==2.14.0`. That version of Keras is then available via both import keras and from tensorflow import keras (the `tf.keras` namespace).

Starting with TensorFlow 2.16, doing `pip install tensorflow` will install Keras 3. When you have TensorFlow >= 2.16 and Keras 3, then by default `from tensorflow import keras` (`tf.keras`) will be Keras 3.

Meanwhile, the legacy Keras 2 package is still being released regularly and is available on PyPI as `tf_keras` (or equivalently `tf-keras` – note that `-` and `_` are equivalent in PyPI package names). To use it, you can install it via `pip install tf_keras` then import it via `import tf_keras as keras`.


## Files and Directories

- **combined-model**: Command line implementation for easy transfer learning of built-in TensorFlow-Keras models
    - combined-model.py: *Training and prediction mode*

    - utils.py: *Helper utility functions*
 
- **predict**:
    - classify_images.py: *Classify images with pre-trained CNNs*

- **simplified**:
    - simplified_carbonate_classification.py: *Simplified training example using TensorFlow Hub*

- **train**
    - carbonates_training_inceptionv3.py: *Uses a Keras ImageNet pre-trained InceptionV3 model to recognise seven Dunham classes using transfer learning and fine tuning*
      
    - carbonates_training_vgg19.py: *Uses a Keras ImageNet pre-trained VGG19 model to recognise seven Dunham classes using transfer learning and fine tuning*


## Usage
To use the training code, please ensure you have set up your dataset in folders following this structure:

    ├── "dataset_name"                   
    |   ├── class_1
    |   |   ├── class_1_images_1
    |   |   ├── class_1_images_2
    |   |   ├── class_1_images_X
    |   |   ├── .....
    |   ├── class_2
    |   |   ├── class_2_images_1
    |   |   ├── class_2_images_2
    |   |   ├── class_2_images_X
    |   |   ├── .....
    |   ├── class_3
    |   |   ├── class_3_images_1
    |   |   ├── class_3_images_2
    |   |   ├── class_3_images_X
    |   |   ├── .....


## References

* VGG16 and VGG19 - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)
* InceptionV3 - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) (CVPR 2016)
* ResNet - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
* DenseNet - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)


## Acknowledgement
This code is part of Harriet Dawson's PhD work and you can [visit her GitHub repository](https://github.com/harrietldawson) where the primary version of this code resides. The work was carried out under the supervision of [Cédric John](https://github.com/cedricmjohn) and all code from the research group can be found in the [John Lab GitHub repository](https://github.com/johnlab-research).

<a href="https://www.john-lab.org">
<img src="https://www.john-lab.org/wp-content/uploads/2023/01/footer_small_logo.png" style="width:220px">
</a>
