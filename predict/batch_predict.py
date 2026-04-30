"""
Batch inference for the Dunham carbonate facies classifier.

Usage:
    python predict/batch_predict.py --folder /path/to/images
    python predict/batch_predict.py --folder /path/to/images --output results.csv
    python predict/batch_predict.py --folder /path/to/images --weights /path/to/custom.h5
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tf_keras as keras
from tf_keras.preprocessing.image import img_to_array, load_img


class DunhamPredictor:
    CLASSES = [
        "boundstone",
        "floatstone",
        "grainstone",
        "mudstone",
        "packstone",
        "rudstone",
        "wackestone",
    ]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, weights_path: str):
        self.model = keras.models.load_model(
            weights_path,
            custom_objects={"KerasLayer": hub.KerasLayer},
        )

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = load_img(image_path, target_size=(299, 299))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0) / 255.0
        return arr

    def predict_folder(self, folder_path: str, output_csv: str = None) -> pd.DataFrame:
        folder = pathlib.Path(folder_path)
        image_files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        if not image_files:
            print(f"No images found in {folder_path}")
            return pd.DataFrame()

        rows = []
        for img_path in image_files:
            arr = self._preprocess(str(img_path))
            logits = self.model.predict(arr, verbose=0)
            probs = keras.activations.softmax(
                keras.backend.constant(logits)
            ).numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_class = self.CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            row = {
                "filename": img_path.name,
                "predicted_class": pred_class,
                "confidence": round(confidence, 4),
            }
            for cls, score in zip(self.CLASSES, probs):
                row[cls] = round(float(score), 4)
            rows.append(row)
            print(f"{img_path.name}: {pred_class} ({confidence:.1%})")

        df = pd.DataFrame(rows)

        if output_csv is None:
            output_csv = str(folder / "dunham_predictions.csv")
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        return df


def main():
    script_dir = pathlib.Path(__file__).parent
    default_weights = str(script_dir.parent / "weights" / "carbonates_inception_v3_model.h5")

    parser = argparse.ArgumentParser(description="Batch Dunham facies prediction")
    parser.add_argument("--folder", required=True, help="Path to folder of images")
    parser.add_argument("--weights", default=default_weights, help="Path to model weights (.h5)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <folder>/dunham_predictions.csv)")
    args = parser.parse_args()

    predictor = DunhamPredictor(args.weights)
    predictor.predict_folder(args.folder, output_csv=args.output)


if __name__ == "__main__":
    main()
