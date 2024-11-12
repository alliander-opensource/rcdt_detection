# SPDX-FileCopyrightText: Alliander N. V.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from ultralytics import FastSAM
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
import numpy as np

SEGMENTATION_MODEL_PATH: str = str(Path.home() / "models" / "FastSAM-x.pt")


def load_segmentation_model(model_path: str = SEGMENTATION_MODEL_PATH) -> Model:
    """Load segmentation model from given path."""
    return FastSAM(model_path)


def segment_image(model: Model, image: np.ndarray) -> Results:
    """Segment given image using given model."""
    height, width, _ = image.shape
    return model(image, imgsz=(height, width))[0]
