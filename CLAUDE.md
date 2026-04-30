# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transfer learning framework for geological image classification (carbonate facies), used in the Dawson et al., 2023 paper. Classifies thin-section images into seven Dunham carbonate classes: boundstone, floatstone, grainstone, mudstone, packstone, rudstone, wackestone.

## Dependencies

- Python with TensorFlow <= 2.15 + Keras 2 (the code uses `tf.keras` / `tf_keras` syntax, incompatible with Keras 3)
- NumPy, OpenCV (`cv2`), Matplotlib

For TF 2.16+, install legacy Keras 2 via `pip install tf_keras` and import as `import tf_keras as keras`.

## Running the Code

**Training (combined-model):**
```bash
python combined-model/combined-model.py --dataset <path_to_dataset> --model InceptionV3 --num_epochs 25
```
Key args: `--model` (VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, DenseNet121/169/201), `--batch_size`, `--dropout`, `--h_flip`, `--v_flip`, `--rotation`, `--continue_training`.

**Prediction (combined-model):**
```bash
python combined-model/combined-model.py --mode predict --image <path_to_image> --model InceptionV3 --dataset <dataset_name>
```

**Prediction (legacy scripts):**
```bash
python predict/classify_images.py --image <path_to_image> --model vgg19
# or --model inceptionv3
```

**Standalone training scripts:**
```bash
python train/carbonates_training_inceptionv3.py
python train/carbonates_training_vgg19.py
```

## Dataset Structure

```
dataset_name/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Architecture

The `combined-model/` directory is the primary, general-purpose implementation:
- **`combined-model.py`**: CLI entry point for both training and prediction. Selects a pretrained ImageNet backbone, freezes its layers, attaches two FC layers (1024 units each) with dropout, and a softmax output layer. Saves/loads weights to `./checkpoints/`.
- **`utils.py`**: Helpers — `build_finetune_model()` (freezes base, appends FC+dropout+softmax), `save_class_list()`/`load_class_list()`, `get_num_files()`, `plot_training()`.

The `predict/` and `train/` directories contain earlier, model-specific scripts (VGG19, InceptionV3) that preceded the unified `combined-model` approach. The `simplified/` directory contains a TensorFlow Hub-based simplified example.

The `weights/` directory (untracked) stores pre-trained model weight files (`.h5`).
