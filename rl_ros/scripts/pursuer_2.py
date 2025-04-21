#!/usr/bin/env python3
import ray 
import torch
import os
import yaml
import rclpy
import numpy as np
import numpy as np
import pathlib
import mavros

from typing import Dict, Any, List
from ray import tune
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModule
from re import S

from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.utils.mask import SimpleEnvMaskModule
from jarvis.envs.simple_agent import PlaneKinematicModel
from jarvis.algos.pro_nav import ProNavV2

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from mavros.base import SENSOR_QOS
from ros_mpc.rotation_utils import (ned_to_enu_states,
                                    get_quaternion_from_euler,
                                    euler_from_quaternion,
                                    convert_enu_state_sol_to_ned)

def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

"""
Refere to the following for inference code:
https://github.com/ray-project/ray/blob/master/rllib/examples/inference/policy_inference_after_training.py
https://docs.ray.io/en/latest/rllib/getting-started.html

Make an option to where you can use PN or PPO 
- Subscribe to mavros position -> this will be the evader or good guy
- From that position set the intial position based on an offset 
- Then when entering the main thread we will pursue the evader
- If using PN:
    - Just calculate relative heading commands and send them to the evader
    
"""

X_IDX = 0
Y_IDX = 1
Z_IDX = 2
ROLL_IDX = 3
PITCH_IDX = 4
YAW_IDX = 5
VX_IDX = 6
VY_IDX = 7
VZ_IDX = 8

# Used to clean up the Ray processes after training
ray.shutdown()
# For debugging purposes
# ray.init(local_mode=True)
ray.init()
pkg_path = get_package_share_directory('rl_ros')
config_dir = os.path.join(pkg_path, 'config')
config_file = os.path.join(config_dir, 'simple_env_config.yaml')

# Load your environment configuration (same as used in training).
env_config = load_yaml_config(
    config_file)['battlespace_environment']

# -------------------------
# Inference Code Starts Here
# -------------------------
tune.register_env("pursuer_evader_env", lambda config:
                  create_multi_agent_env(config=config,
                                         env_config=env_config))


class PursuerNode(Node):
    def __init__(self, ns='',
                 use_pn: bool = True):
        super().__init__('pursuer_node')
        
        # make params to set offset positions
        self.declare_parameter('offset_x', -200.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', 45.0) 
        self.declare_parameter('offset_yaw_dg', 0.0) 
        self.offset_x: float = self.get_parameter('offset_x').value
        self.offset_y: float = self.get_parameter('offset_y').value
        self.offset_z: float = self.get_parameter('offset_z').value
        self.offset_yaw_dg: float = self.get_parameter('offset_yaw_dg').value
        self.target_enu: np.ndarray = np.zeros(7)
        self.enu_state: np.ndarray = np.zeros(7)
        self.max_vel: float = 30.0

        # Need to flag this with pursuer id number 
        self.use_pn:bool = use_pn
        
        if self.use_pn:
            self.pronav: ProNavV2 = ProNavV2()
        else:
            self.pronav = None
            
        self.plane_model: PlaneKinematicModel = PlaneKinematicModel()
        self.start_spawn: bool = False
        self.target_sub = self.create_subscription(mavros.local_position.Odometry,
                                                  'mavros/local_position/odom',
                                                  self.target_sub_callback,
                                                  qos_profile=SENSOR_QOS)
        self.pursuer_str: str = '/pursuer_2'
        self.odom_topic:str = "/pursuer_2/odometry"
        self.scaled_odom_topic:str = "/pursuer_2/scaled_odometry"
        self.pursuer_odometry_pub: Publisher = self.create_publisher(
            Odometry, self.odom_topic, 10)
        self.scaled_pursuer_odometry_pub: Publisher = self.create_publisher(
            Odometry, self.scaled_odom_topic, 10)
        self.scale_value:int = 50
        self.timer_value: float = 0.05
        # self.timer = self.create_timer(self.timer_value, self.step)
        # self.get_logger().info("Pursuer node started")

    def spawn_pursuer(self) -> None:
        """
        Spawn the pursuer
        """
        # self.enu_state[X_IDX] = self.target_enu[X_IDX] + self.offset_x
        # self.enu_state[Y_IDX] = self.target_enu[Y_IDX] + self.offset_y
        # self.enu_state[Z_IDX] = self.target_enu[Z_IDX] + self.offset_z
        self.enu_state[X_IDX] = self.offset_x
        self.enu_state[Y_IDX] = self.offset_y
        self.enu_state[Z_IDX] = self.offset_z
        self.enu_state[YAW_IDX] = self.target_enu[YAW_IDX] + np.deg2rad(self.offset_yaw_dg)
        self.enu_state[VX_IDX] = self.max_vel
        self.plane_model.state_info = self.enu_state
        self.get_logger().info(f"Spawned pursuer at {self.enu_state[X_IDX]}, {self.enu_state[Y_IDX]}, {self.enu_state[Z_IDX]}")
        self.start_spawn = True

    def target_sub_callback(
        self, msg: mavros.local_position.Odometry) -> None:
        """
        Get the target position
        Compute the command trajectory 
        Step the model 
        Publish the position of the pursuer
        """
        self.target_enu[0] = msg.pose.pose.position.x
        self.target_enu[1] = msg.pose.pose.position.y
        self.target_enu[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)

        self.target_enu[3] = roll
        self.target_enu[4] = pitch
        self.target_enu[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        # get magnitude of velocity
        self.target_enu[6] = np.sqrt(vx**2 + vy**2 + vz**2)
        
        if not self.start_spawn:
            self.spawn_pursuer()
        else:
            self.step()
        
    def update_model_state(self) -> None:
        """
        Get the model state
        """
        # self.enu_state = self.plane_model.get_state()
        state_info:np.ndarray = self.plane_model.state_info
        if state_info is None:
            self.get_logger().info("State info is None")
            self.spawn_pursuer()
            
        self.enu_state[X_IDX] = state_info[0]
        self.enu_state[Y_IDX] = state_info[1]
        self.enu_state[Z_IDX] = state_info[2]
        self.enu_state[ROLL_IDX] = state_info[3]
        self.enu_state[PITCH_IDX] = state_info[4]
        self.enu_state[YAW_IDX] = state_info[5]
        self.enu_state[VX_IDX] = state_info[6]
        
    def get_commands(self) -> np.ndarray:
        """
        Compute the trajectory:
        If using PN:
            - Returns an action command consisting of the following:
                [pitch, yaw_cmd (global), airspeed]
        Otherwise we use the PPO model

        """
        if self.use_pn:

            current_pos:np.ndarray = np.array(self.enu_state[0:3])
            # target_pos:np.ndarray = np.array(self.enu
            target_pos:np.array = np.array(self.target_enu[0:3])
            relative_pos:List[float] = target_pos - current_pos
            distance = np.linalg.norm(relative_pos)
            print(f"Distance to target: {distance}")
            relative_vel: float = self.target_enu[VX_IDX] - self.enu_state[VX_IDX]
            action = self.pronav.predict(
                current_pos=current_pos,
                relative_pos=relative_pos,
                current_heading=self.enu_state[YAW_IDX],
                current_speed=self.enu_state[VX_IDX],
                relative_vel=relative_vel,
                consider_yaw=False,
                max_vel=self.max_vel,
            )
            return action
        else:
            pass
        
    def publish_odometry(self) -> None: 
        """
        Publish the odometry
        """
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link_2'
        msg.pose.pose.position.x = self.enu_state[X_IDX]
        msg.pose.pose.position.y = self.enu_state[Y_IDX]
        msg.pose.pose.position.z = self.enu_state[Z_IDX]
        # quaternion attitudes
        qx, qy, qz, qw = get_quaternion_from_euler(
            roll=self.enu_state[ROLL_IDX],
            pitch=self.enu_state[PITCH_IDX],
            yaw=self.enu_state[YAW_IDX]
        )
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        self.pursuer_odometry_pub.publish(msg)
        
    def publish_scaled_odometry(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link_2'
        msg.pose.pose.position.x = self.enu_state[X_IDX] / self.scale_value
        msg.pose.pose.position.y = self.enu_state[Y_IDX] / self.scale_value
        msg.pose.pose.position.z = self.enu_state[Z_IDX] / self.scale_value
        
        # quaternion attitudes
        qx, qy, qz, qw = get_quaternion_from_euler(
            roll=self.enu_state[ROLL_IDX],
            pitch=self.enu_state[PITCH_IDX],
            yaw=self.enu_state[YAW_IDX]
        )
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self.scaled_pursuer_odometry_pub.publish(msg)
        
    def step(self) -> None:
        """
        Step the model
        """
        # Get the action from the model
        if self.use_pn:
            # self.update_model_state()
            action = self.get_commands()
            # update the model with the action
            next_state = self.plane_model.rk45(
                x=self.enu_state,
                u=action,
                dt=0.05,
            )
            self.plane_model.state_info = next_state
            self.update_model_state()
            # publish the odometry
            self.publish_odometry()
            self.publish_scaled_odometry()
        else:
            pass
    
def create_multi_agent_env(config: Dict[str, Any],
                           env_config: Dict[str, Any]) -> PursuerEvaderEnv:
    return PursuerEvaderEnv(
        config=env_config)


def main(args=None):
    rclpy.init(args=args)
    ray.init(ignore_reinit_error=True)
    # print the pwd
    print("Current working directory:", os.getcwd())
    #env = create_multi_agent_env(config=None, env_config=env_config)
    pursuer_node = PursuerNode()
    
    # # check if all the arrays are all zeros
    if np.all(pursuer_node.target_enu == 0):
        rclpy.spin_once(pursuer_node, timeout_sec=0.1)
        pursuer_node.spawn_pursuer()
    else:
        pursuer_node.get_logger().info("Pursuer already spawned")
        
    while rclpy.ok():
        try:
            rclpy.spin_once(pursuer_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    pursuer_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()