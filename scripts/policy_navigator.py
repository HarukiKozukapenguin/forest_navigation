#!/usr/bin/env python3
print("initialize")
import rospy


from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from aerial_robot_msgs.msg import ObstacleArray
from aerial_robot_msgs.msg import FlightNav
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import Image

import torch
print("torch.cuda.is_available() is ",torch.cuda.is_available())
print("torch.__version__ is ", torch.__version__)
import numpy as np

from scipy.spatial.transform import Rotation as R 
from stable_baselines3.common.utils import get_device
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
# import csv


class AgileQuadState:
    def __init__(self, quad_state,transition):
        # self.t = quad_state.t

        self.pos = np.array([quad_state.pose.pose.position.x-transition[0],
                             quad_state.pose.pose.position.y-transition[1],
                             quad_state.pose.pose.position.z], dtype=np.float32)
        self.att = np.array([quad_state.pose.pose.orientation.x,
                             quad_state.pose.pose.orientation.y,
                             quad_state.pose.pose.orientation.z,
                             quad_state.pose.pose.orientation.w], dtype=np.float32)
        self.vel = np.array([quad_state.twist.twist.linear.x,
                             quad_state.twist.twist.linear.y,
                             quad_state.twist.twist.linear.z], dtype=np.float32)
        self.omega = np.array([quad_state.twist.twist.angular.x,
                               quad_state.twist.twist.angular.y,
                               quad_state.twist.twist.angular.z], dtype=np.float32)


class AgilePilotNode:
    def __init__(self, vision_based=False):
        print("Initializing agile_pilot_node...")
        rospy.init_node('policy_navigator', anonymous=False)

        self.vision_based = vision_based
        self.ppo_path = rospy.get_param("~ppo_path")
        # self.real_transition = rospy.get_param("~real_transition")
        # print(self.real_transition)
        self.publish_commands = False
        self.state = None
        self.goal_lin_vel = np.array([5,0,0],dtype="float32")
        self.world_box = np.array([-0.3, 4.2 ,-1.5, 1.5, 0.2, 2.0],dtype="float32")
        # should change when changing position
        x = rospy.get_param("~shift_x")
        y = rospy.get_param("~shift_y")
        self.initial_position = np.array([x, y],dtype="float32")
        self.learned_world_box = np.array([-10, 70 ,-1.5, 1.5, 0.2, 2.0],dtype="float32")
        self.rl_policy = None
        # should change depending on world flame's origin

        quad_name = 'multirotor'

        
        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)
        self.stop_sub = rospy.Subscriber("/" + quad_name + "/stop_navigation", Empty, self.stop_callback,
                                          queue_size=1, tcp_nodelay=True)
        

        # Observation subscribers
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/uav/cog/odom", Odometry, self.state_callback,
                                         queue_size=1, tcp_nodelay=True)

        self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/polar_pixel", ObstacleArray,
                                             self.obstacle_callback, queue_size=1, tcp_nodelay=True)

        # Command publishers
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/uav/nav", FlightNav,
                                          queue_size=1)
        self.obs_pub = rospy.Publisher("/" + quad_name + "/uav/observation", Float64MultiArray,
                                          queue_size=1)
        self.act_pub = rospy.Publisher("/" + quad_name + "/uav/action", Float64MultiArray,
                                          queue_size=1)
        self.command = FlightNav()
        self.command.target = 1
        self.command.pos_xy_nav_mode = 4
        self.command.pos_z_nav_mode = 4

        self.lstm_states = None

        self.n_act = np.zeros(7)

        print("Initialization completed!")


    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data,self.initial_position)

    def obstacle_callback(self, obs_data):
        if self.vision_based:
            return
        if self.state is None:
            return
        # self.rl_policy = None
        if self.ppo_path is not None and self.rl_policy is None:
            self.rl_policy = self.load_rl_policy(self.ppo_path)
        vel_msg = self.rl_example(state=self.state, obstacles=obs_data, rl_policy=self.rl_policy)

        if self.publish_commands:
            self.linvel_pub.publish(vel_msg)

    def rl_example(self, state, obstacles, rl_policy=None):
        obs_vec = np.array(obstacles.boxel)
        # Convert state to vector observation
        goal_vel = self.goal_lin_vel
        world_box = self.learned_world_box
        att_aray = state.att
        rotation_matrix = R.from_quat(att_aray)
        euler = rotation_matrix.as_euler('xyz')
        rotation_matrix = rotation_matrix.as_matrix().reshape((9,), order="F")

        # print("state.pos[0]: ", state.pos[0])
        # print("self.world_box[1]-1.2: ", self.world_box[1]-1.2)
        goal = self.world_box[1]-1.2<state.pos[0]
        inside_range = True
        for i in range (3):
            inside_range &= self.world_box[i*2]<state.pos[i]<self.world_box[i*2+1]
        if goal or not inside_range:
            if goal:
                print("Goal!")    
            if not inside_range:
                print("Out of range!")
            self.command.target_pos_x = self.world_box[1]+self.initial_position[0]-0.8
            self.command.target_pos_y = 0
            self.command.target_pos_z = state.pos[2]

            self.command.target_vel_x = 0
            self.command.target_vel_y = 0
            self.command.target_vel_z = 0

            # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
            self.command.target_yaw = euler[2]
            return self.command

        policy, obs_mean, obs_var, act_mean, act_std = rl_policy

        normalized_p = np.zeros(3)
        for i in range(3):
            normalized_p[i] = (state.pos[i]-self.learned_world_box[2*i])/(self.learned_world_box[2*i+1]-self.learned_world_box[2*i])

        obs = np.concatenate([
            self.n_act.reshape((7)), goal_vel, rotation_matrix, state.pos, normalized_p, state.vel, obs_vec, state.omega,
            np.array([world_box[2] - state.pos[1], world_box[3] - state.pos[1], 
            world_box[4] - state.pos[2] , world_box[5] - state.pos[2]])
    ], axis=0).astype(np.float64)

        # observation_msg = Float64MultiArray()
        # observation_msg.data = obs.tolist()
        # self.obs_pub.publish(observation_msg)

        # with open('debug.csv', 'a') as f:
            # print("now writing")
            # obs_list = obs.tolist()
            # print(obs_list)

            # writer = csv.writer(f)
            # writer.writerow(obs_list)
        # f.close()

        obs = obs.reshape(-1, obs.shape[0])
        norm_obs = self.normalize_obs(obs, obs_mean, obs_var)
        #  compute action

        # print("type(norm_obs) is ",type(norm_obs))
        # print("norm_obs is ",norm_obs.shape)
        self.n_act, self.lstm_states = policy.predict(norm_obs, state = self.lstm_states, deterministic=True)

        # print(self.n_act)
        
        # action_msg = Float64MultiArray()
        # action_msg.data = (self.n_act[0, :]).tolist()
        # # print(type(action_msg.data))
        # # print(action_msg.data)

        # self.act_pub.publish(action_msg)
        # print("self.n_act",self.n_act)
        # print("self.n_act.shape",self.n_act.shape)
        action = (self.n_act * act_std + act_mean)[0, :]

        # cmd freq is same as simulator? cf. in RL dt = 0.02
        momentum = 0.9
        self.command.target_pos_x = (1-momentum)*(state.pos[0] + action[0]+self.initial_position[0])+momentum*self.command.target_pos_x
        self.command.target_pos_y = (1-momentum)*(state.pos[1] + action[1]+self.initial_position[1])+momentum*self.command.target_pos_y
        self.command.target_pos_z = (1-momentum)*(state.pos[2] + action[2])+momentum*self.command.target_pos_z

        self.command.target_vel_x = float((1-momentum)*(state.vel[0] + action[3])+momentum*self.command.target_vel_x)
        self.command.target_vel_y = float((1-momentum)*(state.vel[1] + action[4])+momentum*self.command.target_vel_y)
        self.command.target_vel_z = float((1-momentum)*(state.vel[2] + action[5])+momentum*self.command.target_vel_z)

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        self.command.target_yaw = (1-momentum)*(euler[2] + action[6])+momentum*self.command.target_yaw

        return self.command
    
    def load_rl_policy(self, policy_path):
        print("============ policy_path: ", policy_path)
        policy_dir = policy_path  + "/policy.pth"
        rms_dir = policy_path + "/rms.npz"

        act_mean = np.array([0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0])[np.newaxis, :] 
        act_std = np.array([0.6, 0.6, 0.3, 1.0, 1.0, 1.0, 0.1])[np.newaxis, :] 

        rms_data = np.load(rms_dir)
        obs_mean = np.mean(rms_data["mean"], axis=0)
        obs_var = np.mean(rms_data["var"], axis=0)

        # # -- load saved varaiables 
        device = get_device("auto")

        # Create policy object
        policy = MlpLstmPolicy.load(policy_dir, device = device)

        return policy, obs_mean, obs_var, act_mean, act_std


    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True
        self.command.target_pos_x = self.state.pos[0]+self.initial_position[0]
        self.command.target_pos_y = self.state.pos[1]+self.initial_position[1]
        self.command.target_pos_z = self.state.pos[2]

        self.command.target_vel_x = self.state.vel[0]
        self.command.target_vel_y = self.state.vel[1]
        self.command.target_vel_z = self.state.vel[2]

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        rotation_matrix = R.from_quat(self.state.att)
        euler = rotation_matrix.as_euler('xyz')
        self.command.target_yaw = euler[2]

    def stop_callback(self, data):
        print("Stay current position!")
        
        self.command.target_pos_x = self.state.pos[0]+self.initial_position[0]
        self.command.target_pos_y = self.state.pos[1]+self.initial_position[1]
        self.command.target_pos_z = self.state.pos[2]

        self.command.target_vel_x = 0
        self.command.target_vel_y = 0
        self.command.target_vel_z = 0

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        rotation_matrix = R.from_quat(self.state.att)
        euler = rotation_matrix.as_euler('xyz')
        self.command.target_yaw = euler[2]
        self.linvel_pub.publish(self.command)
        self.publish_commands = False


    def normalize_obs(self, obs, obs_mean, obs_var):
        return (obs - obs_mean) / np.sqrt(obs_var + 1e-8)


if __name__ == '__main__':
    agile_pilot_node = AgilePilotNode()
    rospy.spin()