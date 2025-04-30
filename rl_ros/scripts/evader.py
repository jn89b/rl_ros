#!/usr/bin/env python3
import ray 
import torch
import os
import yaml
import rclpy
import math
import numpy as np
import pathlib
import mavros

from typing import Dict, Any,List
from ray import tune
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.envs.simple_agent import Evader
from jarvis.utils.mask import SimpleEnvMaskModule
from jarvis.utils.vector import StateVector
from jarvis.envs.multi_agent_hrl import HRLMultiAgentEnv

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from re import S
from jarvis.algos.pro_nav import ProNavV2

from rl_ros.PID import PID, FirstOrderFilter
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from ament_index_python.packages import get_package_share_directory
from mavros.base import SENSOR_QOS
from ros_mpc.rotation_utils import (
    euler_from_quaternion)
from drone_interfaces.msg import CtlTraj
def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

"""
Refere to the following for inference code:
https://github.com/ray-project/ray/blob/master/rllib/examples/inference/policy_inference_after_training.py
https://docs.ray.io/en/latest/rllib/getting-started.html
"""

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

def create_hrl_env(config: Dict[str, Any],
                   env_config: Dict[str, Any]) -> HRLMultiAgentEnv:

    return HRLMultiAgentEnv(
        config=env_config)


def yaw_enu_to_ned(yaw_enu: float) -> float:
    """
    Convert yaw angle from ENU to NED.

    The conversion is symmetric:
        yaw_ned = (pi/2 - yaw_enu) wrapped to [-pi, pi]

    Parameters:
        yaw_enu (float): Yaw angle in radians in the ENU frame.

    Returns:
        float: Yaw angle in radians in the NED frame.
    """
    yaw_ned = np.pi/2 - yaw_enu
    return wrap_to_pi(yaw_ned)


def wrap_to_pi(angle: float) -> float:
    """
    Wrap an angle in radians to the range [-pi, pi].

    Parameters:
        angle (float): Angle in radians.

    Returns:
        float: Angle wrapped to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_relative_ned_yaw_cmd(
        current_ned_yaw: float,
        inert_ned_yaw_cmd: float) -> float:

    yaw_cmd: float = inert_ned_yaw_cmd - current_ned_yaw

    # wrap the angle to [-pi, pi]
    return wrap_to_pi(yaw_cmd)


X_IDX = 0
Y_IDX = 1
Z_IDX = 2
ROLL_IDX = 3
PITCH_IDX = 4
YAW_IDX = 5
VX_IDX = 6
VY_IDX = 7
VZ_IDX = 8

CMD_PITCH_IDX = 0
CMD_YAW_IDX = 1
CMD_AIRSPEED_IDX = 2

OFFENSIVE_IDX = 0
DEFENSIVE_IDX = 1

class EvaderNode(Node):
    
    """
    #TODO: ADD docstring and also failchecks
    Failchecks are :
        - If I'm going too far out of bounds switch to offensive policy
    """
    def __init__(self, ns='',
                 evader_policy: SimpleEnvMaskModule = None,
                 hrl_policy: SimpleEnvMaskModule = None,
                 offensive_policy: SimpleEnvMaskModule = None,
                 env: PursuerEvaderEnv = None) -> None:
        super().__init__('pursuer_node')
        self.init_rl: bool = False
        self.evader_policy = evader_policy
        self.hrl_policy = hrl_policy
        self.offensive_policy = offensive_policy
        self.env = env
        self.use_pn: bool = True
        self.x_bounds: List[float] = [-500, 10]
        self.y_bounds: List[float] = [-500, 400]
        self.z_bounds: List[float] = [40, 80]
        
        if self.use_pn:
            self.pronav: ProNavV2 = ProNavV2()
        else:
            self.pronav = None
        
        self.num_actions = env.action_spaces["0"]["action"].nvec.sum()
        self.evader = self.env.get_evader_agents()[0]
        
        self.state_enu: np.ndarray = np.zeros(7)
        self.state_enu: np.ndarray = np.zeros(7)
        self.velocities: np.ndarray = np.zeros(3)
        
        self.target_location: np.ndarray = np.array(
            [-150, 150, 65.0])
        self.final_z_offset: float = -5.0
        
        self.won:bool = False
        self.win_distance:float = 45.0
        self.pub_traj = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        
        self.odom_sub = self.create_subscription(
            mavros.local_position.Odometry, 
            'mavros/local_position/odom', 
            self.enu_callback, 
            qos_profile=SENSOR_QOS)
        
        self.pursuer_sub = self.create_subscription(
            Odometry, 
            'pursuer/odometry', 
            self.pursuer_callback, 
            qos_profile=SENSOR_QOS)
        
        self.pursuer_2_sub = self.create_subscription(
            Odometry, 
            'pursuer_2/odometry', 
            self.pursuer_two_callback, 
            qos_profile=SENSOR_QOS)
        
        self.policy_pub = self.create_publisher(
            Int32, 'policy', 10)
        
        self.pursuer_relative_states: np.array = np.zeros(5)
        self.pursuer_two_relative_states: np.array = np.zeros(5)
        self.initialized = False
        # Used to switch between two policies
        self.hrl_policy_timer:float = 0.25
        self.start_time: float = self.get_clock().now().nanoseconds/1e9
        self.current_policy:int = None
        self.police_switch:List[int] = [OFFENSIVE_IDX, DEFENSIVE_IDX]
        self.timer_period: float = 0.05
        self.offset_states = np.zeros(7)
        
        self.dz_filter : FirstOrderFilter = FirstOrderFilter(
            tau=0.5, dt=0.025, x0=0.0)
        self.yaw_filter : FirstOrderFilter = FirstOrderFilter(
            tau=0.4, dt=0.025, x0=0.0)
        
        self.dz_controller: PID = PID(
            kp=0.025, ki=0.0, kd=0.01,
            min_constraint=np.deg2rad(-12),
            max_constraint=np.deg2rad(10),
            use_derivative=True,
            dt = 0.025)
        
        self.roll_controller: PID = PID(
            kp=0.25, ki=0.0, kd=0.05,
            min_constraint=np.deg2rad(-45),
            max_constraint=np.deg2rad(45),
            use_derivative=True,
            dt = 0.025)

    def enu_callback(
        self, msg: mavros.local_position.Odometry) -> None:
        """            self.current_policy 
        Get the target position
        Compute the command trajectory 
        Step the model 
        Publish the position of the pursuer
        """
        self.state_enu[0] = msg.pose.pose.position.x
        self.state_enu[1] = msg.pose.pose.position.y
        self.state_enu[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)

        self.state_enu[3] = roll
        self.state_enu[4] = pitch
        self.state_enu[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        # get magnitude of velocity
        self.state_enu[6] = np.sqrt(vx**2 + vy**2 + vz**2)
        self.velocities[0] = vx
        self.velocities[1] = vy
        self.velocities[2] = vz
        
        if not self.initialized:
            # get the offset states
            self.offset_states[0] = self.state_enu[0]
            self.offset_states[1] = self.state_enu[1]
            self.offset_states[2] = self.state_enu[2]
            self.offset_states[3] = self.state_enu[3]
            self.offset_states[4] = self.state_enu[4]
            self.offset_states[5] = self.state_enu[5]
            self.offset_states[6] = self.state_enu[6]
            print("offset states: ", self.offset_states)
            self.initialized = True
            
        self.step()
        
    def pursuer_callback(
        self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        
        # get magnitude of velocity
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        relative_x = self.state_enu[0] - x
        relative_y = self.state_enu[1] - y
        relative_z = self.state_enu[2] - z
        relative_speed = self.state_enu[6] - speed
        relative_heading = self.state_enu[5] - yaw
        # wrap relative heading to -pi to pi
        relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi
        self.pursuer_relative_states = np.array([
            relative_x,
            relative_y,
            relative_z,
            relative_speed,
            relative_heading
        ])
        
        # observations: Dict[str, Any] = self.get_evader_observation()
        # action:np.array = self.get_evader_action(observations)
        # self.publish_traj(action)
        self.step()
        
    def pursuer_two_callback(
        self, msg: Odometry) -> None:
        """
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        
        # get magnitude of velocity
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        relative_x = self.state_enu[0] - x
        relative_y = self.state_enu[1] - y
        relative_z = self.state_enu[2] - z
        relative_speed = self.state_enu[6] - speed
        relative_heading = self.state_enu[5] - yaw
        # wrap relative heading to -pi to pi
        relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi
        self.pursuer_two_callback = np.array([
            relative_x,
            relative_y,
            relative_z,
            relative_speed,
            relative_heading
        ])

    def update_evader_in_env(self) -> None:
        self.evader.state_vector.x = self.state_enu[X_IDX]
        self.evader.state_vector.y = self.state_enu[Y_IDX]
        self.evader.state_vector.z = self.state_enu[Z_IDX]
        self.evader.state_vector.roll_rad = self.state_enu[ROLL_IDX]
        self.evader.state_vector.pitch_rad = self.state_enu[PITCH_IDX]
        self.evader.state_vector.yaw_rad = self.state_enu[YAW_IDX]
        self.evader.state_vector.speed = self.state_enu[VX_IDX]        
        self.evader.simple_model.state_info = self.state_enu
        
    def get_high_level_observation(self) -> Dict[str, Any]:
        """
        Returns the high level observation which consists of :
        - Euclidean distance to the target
        - Relative eucildeant distance for each pursuer
        """
        distance_from_goal = np.linalg.norm(
            self.state_enu[:3] - self.target_location)
        
        # get the relative distance to the pursuer
        relative_distance = np.linalg.norm(
            self.state_enu[:3] - self.pursuer_relative_states[:3])
        
        # combine into an array
        high_level_obs = np.array([
            distance_from_goal,
            relative_distance,
            relative_distance
        ], dtype=np.float32)
        num_actions:int = 2
        valid_actions = np.ones(num_actions, dtype=np.float32)
        observations:Dict[str, Any] = {
            "observations": high_level_obs,
            "action_mask": valid_actions
        }
        return observations
        
    def get_evader_observation(self) -> Dict[str, Any]:
        """
        """
        obs = [
            self.state_enu[X_IDX],
            self.state_enu[Y_IDX],
            self.state_enu[Z_IDX],
            self.state_enu[ROLL_IDX],
            self.state_enu[PITCH_IDX],
            self.state_enu[YAW_IDX],
            self.state_enu[VX_IDX], 
            self.velocities[0],
            self.velocities[1],
            self.velocities[2]
        ]
         
        distance = np.linalg.norm(
            self.pursuer_relative_states[:3])
        if distance <= 20.0:
            print("Captured: ", distance)
        else:
            print("Not captured: ", distance)
        #print("pursuer_relative_states: ", self.pursuer_relative_states)
        # include the pursuer positons
        obs = np.concatenate(
            (obs, self.pursuer_relative_states), axis=0)
        obs = np.concatenate(
            (obs, self.pursuer_two_relative_states), axis=0)
                
        # Get the action relative to the pursuer 
        obs = np.array(obs, dtype=np.float32)
        obs[:3] = obs[:3] - self.offset_states[:3]
        action_mask:np.array = self.env.get_action_mask(
            agent=self.evader, 
            action_space_sum=self.num_actions
        )
        return {"observations": obs, "action_mask": action_mask}

    def get_pursuer_observation(self) -> Dict[str, Any]:
        """
        This observation wil
        """
        obs = [
            self.state_enu[X_IDX],
            self.state_enu[Y_IDX],
            self.state_enu[Z_IDX],
            self.state_enu[ROLL_IDX],
            self.state_enu[PITCH_IDX],
            self.state_enu[YAW_IDX],
            self.state_enu[VX_IDX], 
            self.velocities[0],
            self.velocities[1],
            self.velocities[2]
        ]
        # obs[:3] = obs[3:] - self.offset_states[3:]
        relative_pos: np.array = self.state_enu[:3] - \
            self.target_location[:3]
        relative_vel: np.array = self.state_enu[VX_IDX]
        
        los = np.arctan2(
            relative_pos[1], relative_pos[0])
        relative_heading = self.state_enu[YAW_IDX] - los
        # wrap relative heading to -pi to pi
        relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi
        relative_info: List[float] = [
            relative_pos[0], relative_pos[1], relative_pos[2],
            relative_vel, relative_heading]
        obs = np.concatenate([obs, relative_info]).astype(np.float32)
        action_mask:np.array = self.env.get_action_mask(
            agent=self.evader, 
            action_space_sum=self.num_actions
        )
        return {"observations": obs, "action_mask": action_mask}
    
    def get_evader_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Get the action from the model
        Returns the action in array where indexing is 
        0 - pitch 
        1 - yaw
        2 - airspeed
        """
        # Convert the observation to a tensor
        torch_obs_batch = {k: torch.from_numpy(
            np.array([v])) for k, v in obs.items()}
        action_logits = self.evader_policy.forward_inference({"obs": torch_obs_batch})[
            "action_dist_inputs"]
        action_logits = action_logits.detach().numpy().squeeze()
        unwrapped_action: Dict[str, np.array] = self.env.unwrap_action_mask(
            action_logits)
        
        # this is 
        discrete_actions = []
        for k, v in unwrapped_action.items():
            v = torch.from_numpy(v)
            best_action = torch.argmax(v).numpy()
            discrete_actions.append(best_action)
        discrete_actions = np.array(discrete_actions)
        continous_actions = self.env.discrete_to_continuous_action(
            discrete_actions
        )
        
        return continous_actions

    def get_pursuer_action(self, obs: Dict[str, Any]) -> np.ndarray:
        if self.use_pn:
            current_pos:np.ndarray = np.array(self.state_enu[0:3])
            # target_pos:np.ndarray = np.array(self.enu
            target_pos:np.ndarray = np.array(self.target_location[0:3])
            relative_pos:List[float] = target_pos - current_pos
            distance = np.linalg.norm(relative_pos)
            print(f"Distance to target: {distance}")
            if distance <= self.win_distance:
                print("Won the game")
                self.won = True
            
            #relative_vel: float = self.target_location[VX_IDX] - self.enu_state[VX_IDX]
            relative_vel: float = self.state_enu[VX_IDX]
            max_vel:float = 30.0
            action = self.pronav.predict(
                current_pos=current_pos,
                relative_pos=relative_pos,
                current_heading=self.state_enu[YAW_IDX],
                current_speed=self.state_enu[VX_IDX],
                relative_vel=relative_vel,
                consider_yaw=True,
                max_vel=max_vel,
            )
            return action
        else:
            pass

    def publish_traj(self,
                     actions: np.ndarray) -> None:
        """
        Trajectory published must be in NED frame
        Yaw control must be sent as relative NED command
        """
        # Solutions unpacked are in ENU frame
        # we need to convert to NED frame

        ned_yaw_rad = yaw_enu_to_ned(actions[CMD_YAW_IDX])
        ned_yaw_state = yaw_enu_to_ned(self.state_enu[YAW_IDX]) 
        rel_yaw_cmd:float = get_relative_ned_yaw_cmd(
            ned_yaw_state, ned_yaw_rad)
        # rel_yaw_cmd = np.clip(
        #     rel_yaw_cmd, -np.deg2rad(45), np.deg2rad(45))
        rel_yaw_cmd = self.yaw_filter.filter(
            rel_yaw_cmd)
        # relative yaw command is already computed as error 
        # so we set setpoint to 0.0
        if self.roll_controller.prev_error is None:
            self.roll_controller.prev_error = 0.0
            
        roll_cmd = self.roll_controller.compute(
            setpoint=rel_yaw_cmd,
            current_value=0.0,
            dt=0.05
        )
        
        # make sure the roll command has the same sign convention as 
        # the yaw command
        if rel_yaw_cmd < 0.0 and roll_cmd > 0.0:
            roll_cmd = -roll_cmd
        elif rel_yaw_cmd > 0.0 and roll_cmd < 0.0:
            roll_cmd = -roll_cmd
            
        # kp:float = 0.25
        # roll_cmd:float = kp * rel_yaw_cmd
        roll_cmd = np.clip(roll_cmd, -np.deg2rad(45), np.deg2rad(45))
                 
        #dz:float = ref_height - self.state_enu[Z_IDX]
        dz = actions[0]
        # dz is already computed in the model so set setpoint as dz
        # and the current value as 0.0
        dz = self.dz_filter.filter(dz)
        dz = np.clip(dz, -10.0, 10.0)
        if self.dz_controller.prev_error is None:
            self.dz_controller.prev_error = 0.0
            
        pitch_cmd:float = self.dz_controller.compute(
            setpoint=dz,
            current_value=0.0,
            dt=0.05
        )
        pitch_cmd = np.clip(pitch_cmd, -np.deg2rad(12), np.deg2rad(10))
        
        # airspeed command
        airspeed_error = actions[CMD_AIRSPEED_IDX] - self.state_enu[6]
        kp_airspeed:float = 0.25
        airspeed_cmd:float = kp_airspeed * airspeed_error
        min_thrust:float = 0.35
        max_thrust:float = 0.65
        thrust_cmd:float = np.clip(
            airspeed_cmd, min_thrust, max_thrust)
        
        if self.won:
            thrust_cmd = 0.5
        dz = float(dz)
        trajectory: CtlTraj = CtlTraj()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.roll = [roll_cmd, roll_cmd]
        trajectory.pitch = [pitch_cmd, pitch_cmd]
        trajectory.yaw = [rel_yaw_cmd, rel_yaw_cmd]
        trajectory.thrust = [thrust_cmd, thrust_cmd]
        trajectory.z = [dz, dz, dz]
        trajectory.idx = int(0)
        
        self.pub_traj.publish(trajectory)
        
        # publish the policy
        policy_msg = Int32()
        policy_msg.data = self.current_policy
        self.policy_pub.publish(policy_msg)

    def get_high_level_action(self) -> int:
        """
        Get the high level action from the HRL model
        """
        # get the action from the model
        obs = self.get_high_level_observation()
        torch_obs_batch = {k: torch.from_numpy(
            np.array([v])) for k, v in obs.items()}
        action_logits = self.hrl_policy.forward_inference(
            {"obs": torch_obs_batch})["action_dist_inputs"]
        discrete_idx = torch.argmax(action_logits, dim=-1).item()
        discrete_actions = discrete_idx
        
        return discrete_actions
        
    def step(self) -> None:
        """
        Step the model
        We're going to need to use the HRL to pick a policy:
        - Offensive
        - Defensive
        
        """
        self.update_evader_in_env()
        
        self.curent_time = self.get_clock().now().nanoseconds/1e9
        # convert to seconds
        elapsed_time: float = self.curent_time - self.start_time
        # Need to update the timer to switch between policies
        if self.won:
            self.target_location[0] = -100.0
            self.target_location[1] = 0.0
            self.target_location[2] = 60.0
            self.current_policy = OFFENSIVE_IDX
            observation = self.get_pursuer_observation()
            action = self.get_pursuer_action(observation)
            self.publish_traj(action)
            print("Won the game")
            return
        
        if self.state_enu[0] < self.x_bounds[0] or \
            self.state_enu[0] > self.x_bounds[1] or \
            self.state_enu[1] < self.y_bounds[0] or \
            self.state_enu[1] > self.y_bounds[1] or \
                self.state_enu[2] < self.z_bounds[0] or \
                    self.state_enu[2] > self.z_bounds[1]:
                self.current_policy = OFFENSIVE_IDX
                observation = self.get_pursuer_observation()
                action = self.get_pursuer_action(observation)
                self.publish_traj(action)
                print("Out of bounds", 
                      self.state_enu[0], self.state_enu[1])
                return
        
        if self.current_policy is None or elapsed_time >= self.hrl_policy_timer:
            self.current_policy = self.get_high_level_action()
            self.start_time = self.get_clock().now().nanoseconds/1e9
            print("switched to policy: ", self.current_policy)    
        else:
            if self.current_policy == OFFENSIVE_IDX:
                # we will use the offensive policy
                observation = self.get_pursuer_observation()
                action = self.get_pursuer_action(observation)
                print("Offensive policy")
            elif self.current_policy == DEFENSIVE_IDX:
                print("Defensive policy")
                observation = self.get_evader_observation()
                action = self.get_evader_action(observation)
    
            self.publish_traj(action)                
        
        return 

def main(args=None) -> None:
    """
    Evader needs :
    - to subscribe to position of pursuers
    - format into observation and then feed into the model
    - get the action from the model 
    - convert the discrete action to continuous action 
    - publish the action to the evader as trajectory 
    - NEED to make sure coordinate system is correct
    
    Observation must be a dictionary that consists of:
    observations: {
        - x: The x position of the agent
        - y: The y position of the agent
        - z: The z position of the agent
        - roll: The roll of the agent
        - pitch: The pitch of the agent
        - yaw: The yaw of the agent
        - speed: The speed of the agent
        - vx: The x velocity of the agent
        - vy: The y velocity of the agent
        - vz: The z velocity of the agent
        For n other agents in the environment we include the relative positions
        - relative_x: The relative x position of the agent to the other agents
        - relative_y: The relative y position of the agent to the other agents
        - relative_z: The relative z position of the agent to the other agents
        - relative speed: The relative speed of the agent to the other agents
        - relative headings: The relative heading of the agent to the other agents
    }
    action_mask: {
    }
    """
    rclpy.init(args=args)
    ray.init(ignore_reinit_error=True)
    # print the pwd
    print("Current working directory:", os.getcwd())

    env = create_multi_agent_env(config=None, env_config=env_config)
    #env = create_hrl_env(config=None, env_config=env_config)

    # checkpoint_path = os.path.join(
    #     pkg_path, 'rl_weights', 'checkpoint_000179')
    # checkpoint_path = "/develop_ws/src/rl_ros/rl_ros/rl_weights/PPO_hrl/PPO_hrl_env_f5dc8_00000_0_2025-03-02_03-30-20/checkpoint_000179"
    # check if the path exists
    # checkpoint_path = os.path.join(
    #     pkg_path, 'checkpoints', 'checkpoint_000001')
    checkpoint_path = os.path.join(
        pkg_path, 'rl_weights', 'pursuer_evader')
    # load the model from our checkpoints
    # Create only the neural network (RLModule) from our checkpoint.
    # evader_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
    #     pathlib.Path(checkpoint_path) /
    #     "learner_group" / "learner" / "rl_module"
    # )["evader_policy"]
    
    checkpoint_path = os.path.join(
        pkg_path, 'rl_weights', 'HRL_PPO')
    policies: MultiRLModuleSpec = RLModule.from_checkpoint(
        pathlib.Path(checkpoint_path) /
        "learner_group" / "learner" / "rl_module"
    )
    
    print("policy keys: ", policies.keys())
    hrl_policy = policies["good_guy_hrl"]
    offensive_policy = policies["good_guy_offensive"]
    evader_policy = policies["good_guy_defensive"]

    print("Loaded the model from checkpoint")
    evader_node = EvaderNode(ns='evader', 
                             hrl_policy=hrl_policy,
                             offensive_policy=offensive_policy,
                             evader_policy=evader_policy,
                             env=env)
    
    if np.all(evader_node.state_enu == 0):
        rclpy.spin_once(evader_node, timeout_sec=0.1)
    
    while rclpy.ok():
        try:
            rclpy.spin_once(evader_node, timeout_sec=0.1)
        except KeyboardInterrupt:
            break
        
if __name__ == "__main__":
    main()