#!/usr/bin/env python3
import rclpy
import math
import numpy as np

from rclpy.node import Node
from rclpy.duration import Duration
from typing import List
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from nav_msgs.msg import Odometry
from ros2_template_package.config import GOAL_STATE
from rclpy.timer import Rate
from rclpy.timer import Timer


class TargetEstimator(Node):
    """
    Target estimator node
    Used to approximate the target position based on the current position and velocity
    """

    def __init__(self, ns=''):
        super().__init__('target_estimator')

        # https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
        self.target_publisher: Publisher = self.create_publisher(
            Odometry, 'target_position', 10)

        self.target_location: List[float] = GOAL_STATE
        self.timer_period: float = 0.05
        self.timer = self.create_timer(
            self.timer_period, self.publish_target)

    def publish_target(self) -> None:
        """
        """
        msg = Odometry()
        std_dev:float = 2.0
        msg.pose.pose.position.x = self.target_location[0] + np.random.normal(0, std_dev)
        msg.pose.pose.position.y = self.target_location[1] + np.random.normal(0, std_dev)
        msg.pose.pose.position.z = self.target_location[2] + np.random.normal(0, std_dev)
        self.target_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    target_estimator = TargetEstimator()

    while rclpy.ok():
        try:
            rclpy.spin_once(target_estimator, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    target_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
