#!/usr/bin/env python3
from re import S
import rclpy
import math
import numpy as np

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.subscription import Subscription
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ros2_template_package.example_class import Drone


class SubExample(Node):
    def __init__(self, ns=''):
        super().__init__('pub_example')

        self.some_publisher: Subscription = self.create_subscription(
            String, 'adele', self.listen_callback, 10)

        self.timer_period: float = 0.5
        # self.timer = self.create_timer(
        #     self.timer_period, self.listen_callback)

    def listen_callback(self, msg: String) -> None:
        self.get_logger().info(f"Received: {msg.data}")
        line_2: str = "I was wondering if after all these years you'd like to meet"
        self.get_logger().info(f"Sending: {line_2}")


def main(args=None):
    rclpy.init(args=args)
    sub_example = SubExample()
    # drone = Drone(name='jake')
    while rclpy.ok():
        try:
            rclpy.spin_once(sub_example, timeout_sec=0.5)
            # drone.go_brr()

        except KeyboardInterrupt:
            break

    sub_example.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
