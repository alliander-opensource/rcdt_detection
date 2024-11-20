#!/usr/bin/env python3

# SPDX-FileCopyrightText: Alliander N. V.
#
# SPDX-License-Identifier: Apache-2.0

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rcdt_detection_msgs.srv import SegmentImage
from rcdt_detection.segmentation import segment_image, load_segmentation_model
from rcdt_detection.image_manipulation import (
    cv2_image_to_ros_image,
    ros_image_to_cv2_image,
)


class SegmentImageNode(Node):
    def __init__(self) -> None:
        super().__init__("segment_image")
        self.create_service(SegmentImage, "segment_image", self.callback)

        self.pub_input = self.create_publisher(Image, "~/input", 10)
        self.pub_output = self.create_publisher(Image, "~/output", 10)
        self.model = load_segmentation_model("SAM2")

    def callback(
        self, request: SegmentImage.Request, response: SegmentImage.Response
    ) -> SegmentImage.Response:
        im_input = request.image
        encoding = im_input.encoding

        if request.publish:
            self.pub_input.publish(im_input)

        self.get_logger().info("Start segmentation...")
        result = segment_image(self.model, ros_image_to_cv2_image(im_input))
        self.get_logger().info("Finished segmentation!")

        if request.publish:
            im_output = result.plot(labels=False, boxes=False, conf=False)
            self.pub_output.publish(cv2_image_to_ros_image(im_output, encoding))

        response.success = True
        return response


def main(args: str = None) -> None:
    rclpy.init(args=args)
    node = SegmentImageNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
