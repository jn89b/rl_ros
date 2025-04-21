#!/usr/bin/env python3
from re import S
import rclpy
import math
import numpy as np

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class PubExample(Node):
    def __init__(self, ns=''):
        super().__init__('pub_example')

        self.some_publisher: Publisher = self.create_publisher(
            String, 'adele', 10)
        self.timer_period: float = 0.5
        self.timer = self.create_timer(
            self.timer_period, self.publish_message)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, it's me!"
        self.some_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    pub_example = PubExample()

    while rclpy.ok():
        try:
            rclpy.spin_once(pub_example, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    pub_example.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
