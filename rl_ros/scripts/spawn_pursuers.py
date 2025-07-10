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



    
def create_multi_agent_env(config: Dict[str, Any],
                           env_config: Dict[str, Any]) -> PursuerEvaderEnv:
    return PursuerEvaderEnv(
        config=env_config)


class SpawnPursuers(Node):
    def __(self, ns='',
           use_pn:bool = True) -> None:
        super().__init__(ns)
        
        # spawn multiple pursuers
        self.num_pursuers:int = 2
        
        self.use_pn:bool = use_pn
        if self.use_pn:
            self.pronav: ProNavV2 = ProNavV2()
        else:
            self.pronav = None
            
        self.target_sub = self.create_subscription(mavros.local_position.Odometry,
                                                  'mavros/local_position/odom',
                                                  self.target_sub_callback,
                                                  qos_profile=SENSOR_QOS)
        self.pursuer_odometry_pub: Publisher = self.create_publisher(
            Odometry, '/pursuer/odometry', 10)
        self.scaled_pursuer_odometry_pub: Publisher = self.create_publisher(
            Odometry, '/pursuer/scaled_odometry', 10)
        self.scale_value:int = 50
        self.timer_value: float = 0.05
            
    def init_pursuers(self):
        # spawn the pursuers
        self.pursuers = []
        for i in range(self.num_pursuers):
            pursuer = PlaneKinematicModel()
            self.pursuers.append(pursuer)
            self.create_pursuer_publisher(i)
            self.get_logger().info(f"Spawned pursuer {i}")
        
    def create_pursuer_publisher(self, pursuer_id:int) -> None:
        pursuer_topic = f'/pursuer_{pursuer_id}/odometry'
        # create the pursuer publisher
        self.pursuer_pub = self.create_publisher(
            Odometry, pursuer_topic, 10)
        

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