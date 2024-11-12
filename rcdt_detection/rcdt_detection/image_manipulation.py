# SPDX-FileCopyrightText: Alliander N. V.
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

cv_bridge = CvBridge()


def ros_image_to_cv2_image(
    image_message: Image, desired_encoding: str = "passthrough"
) -> np.ndarray:
    """Convert ROS image message to cv2 image."""
    return cv_bridge.imgmsg_to_cv2(image_message, desired_encoding=desired_encoding)


def segmentation_mask_to_binary_mask(mask: torch.Tensor) -> np.ndarray:
    """Convert given mask to np.array with range [0, 255], dtype=uint8, and dimensions [height, width, channels]."""
    binary_mask = mask.data.cpu().numpy().astype(np.uint8)
    binary_mask = binary_mask * 255
    binary_mask = binary_mask.transpose(1, 2, 0)
    return binary_mask


def single_to_three_channel(image: np.array) -> np.ndarray:
    """Convert given single-channel image to three-channel image."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
