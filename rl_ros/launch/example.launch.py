#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    some_namespace = LaunchConfiguration('some_namespace',
                                         default='some_namespace')

    some_node = Node(
        package='ros2_template_package',
        namespace=some_namespace,
        executable='pub_example.py'
    )
    launch_description = LaunchDescription(
        [
            some_node
        ]
    )

    return launch_description
