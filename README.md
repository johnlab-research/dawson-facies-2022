# Transfer Learning for Geological Images

## Description
The code in this repository can be used to easily perform transfer learning using any of the built-in TensorFlow-Keras image classification models used in the Dawson et al., 2023 paper.

## Installation

Requires **Python 3.9–3.12**. The repo ships with a `.python-version` file pointing to the `dawson` pyenv virtualenv.

```bash
pyenv virtualenv 3.12.9 dawson
pyenv local dawson
pip install -r requirements.txt
```

### TensorFlow / Keras compatibility note

From TensorFlow 2.0 to 2.15, `pip install tensorflow` also installs the matching Keras 2 (available as `tf.keras`).

Starting with TensorFlow 2.16, `pip install tensorflow` installs Keras 3 and `tf.keras` becomes Keras 3. This codebase relies on the legacy Keras 2 API, so **TF 2.16+ requires the `tf-keras` package** (installed via `requirements.txt`). Import it as `import tf_keras as keras`.

The pre-trained weights (`weights/carbonates_inception_v3_model.h5`) were built with TensorFlow Hub and require `tensorflow-hub` to load.


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


## Batch Inference

Run the pre-trained InceptionV3 model on a folder of images:

```bash
# Predict all images in a folder (saves dunham_predictions.csv inside the folder)
python predict/batch_predict.py --folder /path/to/images

# Specify a custom output CSV path
python predict/batch_predict.py --folder /path/to/images --output results.csv

# Use custom weights
python predict/batch_predict.py --folder /path/to/images --weights /path/to/custom.h5
```

Results are saved as a CSV with one row per image containing the predicted Dunham class, the top-class confidence score, and individual probability scores for all seven classes (boundstone, floatstone, grainstone, mudstone, packstone, rudstone, wackestone).

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
