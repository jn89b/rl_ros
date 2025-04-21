#!/usr/bin/env python3
import numpy as np
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node

"""
Uses the rviz_drone package to visualize all the trajectories of the drone
"""

def generate_launch_description() -> LaunchDescription:
    
    rate_str = '30.0'
    rate_val = float(30.0)
    
    ### Actual Frames
    actual_fix_wing_broadcaster = Node(
        package='rviz_drone',
        executable='aircraft_actual_frame.py',
        name='aircraft_actual_frame',
        output='screen',
        parameters=[
            {'x': 0.0},
            {'y': 0.0},
            {'z': 0.0},
            {'roll': 0.0},
            {'pitch': 0.0},
            {'yaw': 0.0},
            {'parent_frame': 'map'},
            {'child_frame': 'actual_fixed_wing_frame'},
            {'rate': rate_val}
        ]
        )
    
    
    actual_effector_broadcast_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('pew_pew'),
                'launch',
                'all_effector.launch.py'
            ])
        ),
        # launch_arguments=directional_traj_config.items()
    )
        
    ### VISUALIZERS THAT ARE SCALED DOWN  ###     
    # this visualizes the scaled down frame of the fixed wing 
    scaled_fix_wing_node = Node(
        package='rviz_drone',
        executable='aircraft_frame.py',
        name='aircraft_frame',
        output='screen',
        parameters=[
            {'x': 0.0},
            {'y': 0.0},
            {'z': 0.0},
            {'roll': 0.0},
            {'pitch': 0.0},
            {'yaw': 0.0},
            {'parent_frame': 'map'},
            {'child_frame': 'fixed_wing_frame'},
            {'rate': rate_val}
        ]
        )
    

    #visualizes the scaled down obstacles
    # obs_viz_node = Node(
    #     package='rviz_drone',
    #     executable='obstacle_vis.py',
    #     name='obstacle_visualizer',
    #     output='screen',
    # )
    
    #visualize the goal location 
    goal_viz_node = Node(
        package='ros2_template_package',
        executable='goal_vis.py',
        name='goal_visualizer',
        output='screen',
    )
    
    effector_scaled_broadcast_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('pew_pew'),
                'launch',
                'all_effector_scaled.launch.py'
            ])
        ),
    )
    
    # set the actual frame of the aicraft
    ld = LaunchDescription()
    # actual frame visualizers
    ld.add_action(actual_fix_wing_broadcaster)
    ld.add_action(actual_effector_broadcast_launch)
    
    # scaled down visualizers
    ld.add_action(scaled_fix_wing_node)
    # ld.add_action(obs_viz_node)
    ld.add_action(goal_viz_node)
    
    ld.add_action(effector_scaled_broadcast_launch)
    
    return ld