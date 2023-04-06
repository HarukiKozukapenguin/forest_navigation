#!/usr/bin/env python3
print("initialize")
import rospy


from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from aerial_robot_msgs.msg import ObstacleArray
from aerial_robot_msgs.msg import FlightNav
from sensor_msgs.msg import LaserScan
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
    def __init__(self):
        print("Initializing agile_pilot_node...")
        rospy.init_node('policy_navigator', anonymous=False)

        self.ppo_path = rospy.get_param("~ppo_path")
        self.get_from_hokuyo = rospy.get_param("~hokuyo")
        # self.real_transition = rospy.get_param("~real_transition")
        # print(self.real_transition)
        self.publish_commands = False
        self.stop_navigation = False
        self.state = None
        self.goal_lin_vel = np.array([5,0,0],dtype="float32")
        self.world_box = np.array([-0.3, 25 ,-1.5, 1.5, 0.2, 2.0],dtype="float32")
        # should change when changing position
        x = rospy.get_param("~shift_x")
        y = rospy.get_param("~shift_y")
        self.translation_position = np.array([x, y],dtype="float32")
        self.learned_world_box = np.array([-0.3, 70 ,-1.5, 1.5, 0.2, 2.0],dtype="float32")
        # last value of theta_list is 134 for the range of the quadrotor
        self.theta_list = np.array([5,15,25,35,45,60,75,90, 105, 120, 134])
        self.theta_num = len(self.theta_list)
        self.max_detection_range = 10 #max_detection_range when leraning
        # checked several rosbag, and I found that the normal flight rarely exceeds the tilts over 30 degrees
        self.max_halt_tilt = np.deg2rad(60)
        self.land_tilt = np.deg2rad(40)
        self.rl_policy = None
        # should change depending on world flame's origin

        quad_name = rospy.get_param("~robot_ns")

        self.command = FlightNav()
        self.command.target = 1
        self.command.pos_xy_nav_mode = 4
        self.command.pos_z_nav_mode = 4

        self.lstm_states = None
        self.body_size = rospy.get_param("~body_size")  #radius of quadrotor(0.25~0.5)
        self.collision_distance = 0.10
        self.dist_backup = 0.10
        self.landing_dist_threshold = 0.05
        self.force_landing_dist_threshold = 0.40
        self.beta = 0.002 # min distance for linearization
        learning_max_gain = 10.0
        exec_max_gain = 3.0
        self.vel_conversion = 1/np.sqrt(exec_max_gain/learning_max_gain)
        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)
        self.stop_sub = rospy.Subscriber("/" + quad_name + "/stop_navigation", Empty, self.stop_callback,
                                          queue_size=1, tcp_nodelay=True)
        

        # Observation subscribers
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/uav/cog/odom", Odometry, self.state_callback,
                                         queue_size=1, tcp_nodelay=True)

        if self.get_from_hokuyo:
            self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/scan", LaserScan,
                                                 self.obstacle_callback, queue_size=1, tcp_nodelay=True)
        else:
            self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/polar_pixel", ObstacleArray,
                                                self.obstacle_callback, queue_size=1, tcp_nodelay=True)

        # Command publishers
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/uav/nav", FlightNav,
                                          queue_size=1)
        # self.obs_pub = rospy.Publisher("/" + quad_name + "/uav/observation", Float64MultiArray,
        #                                   queue_size=1)
        self.act_pub = rospy.Publisher("/" + quad_name + "/uav/action", Float64MultiArray,
                                          queue_size=1)
        self.land_pub = rospy.Publisher("/" + quad_name + "/teleop_command" + '/land', Empty, queue_size=1)
        self.force_landing_pub = rospy.Publisher("/" + quad_name + "/teleop_command" + '/force_landing', Empty, queue_size=1)
        self.halt_pub = rospy.Publisher("/" + quad_name + "/teleop_command" + '/halt', Empty, queue_size=1)

        self.n_act = np.zeros(2)

        print("Initialization completed!")


    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data,self.translation_position)

    def obstacle_callback(self, obs_data):
        # obstacle conversion depending on the type of sensor
        # range: 0.0~1.0 [m/self.max_detection_range]
        if self.get_from_hokuyo:
            obs_vec: np.array  = self.LaserScan_to_obs_vec(obs_data)
        else:
            obs_vec: np.array = np.array(obs_data.boxel)
        if self.state is None:
            return
        # self.rl_policy = None

        # when there are bad collision before
        self.calc_tilt()
        if self.is_halt(obs_vec):
            print("Begin halt!")
            self.halt_pub.publish(Empty())
            return
        if self.stop_navigation and self.is_force_landing():
            print("Begin force landing!")
            self.force_landing_pub.publish(Empty())
            return
        if self.stop_navigation and self.is_landing():
            self.landing()
            print("Begin emergency landing!")
            return
        if self.stop_navigation:
            return
        if self.bad_collision(obs_vec):
            self.stop_navigation = True
            self.publish_commands = False
            vel_msg = self.landing_position_setting(obs_vec)
            self.linvel_pub.publish(vel_msg)
            print("Move to emergency landing poisiton!")
            return
        # when there are no bad collision before
        if self.ppo_path is not None and self.rl_policy is None:
            self.rl_policy = self.load_rl_policy(self.ppo_path)
        vel_msg = self.rl_example(state=self.state, obs_vec=obs_vec, rl_policy=self.rl_policy)

        if self.publish_commands:
            # debug to show whith direction quadrotor go in given position
            # print("x_state: ",self.state.pos[0])
            # print("x_direction: ",vel_msg.target_pos_x-(self.state.pos[0]+self.translation_position[0]))
            # print("y_direction: ",vel_msg.target_pos_y-(self.state.pos[1]+self.translation_position[1]))
            self.linvel_pub.publish(vel_msg)

    def rl_example(self, state, obs_vec, rl_policy=None):
        a = -1/self.beta
        b = 1-np.log(self.beta)
        log_obs_vec = np.where(obs_vec < self.beta, a*obs_vec+b, -np.log(obs_vec))
        # obs_vec = np.array(obstacles.boxel)
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
                self.force_landing_pub.publish(Empty())
            self.command.target_pos_x = self.world_box[1]+self.translation_position[0]-0.8
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
            self.n_act.reshape((2)), state.pos[0:2], np.array([state.vel[0]*self.vel_conversion]), np.array([state.vel[1]]), rotation_matrix, state.omega,
            np.array([world_box[2] - state.pos[1], world_box[3] - state.pos[1]]), np.array([self.body_size]), log_obs_vec
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

        print("action: ", action)

        # cmd freq is same as simulator? cf. in RL dt = 0.02
        momentum = 0.0
        self.command.target_pos_x = (1-momentum)*(state.pos[0] + action[0]+self.translation_position[0])+momentum*self.command.target_pos_x
        self.command.target_pos_y = (1-momentum)*(state.pos[1] + action[1]+self.translation_position[1])+momentum*self.command.target_pos_y
        self.command.target_pos_z = 1.0

        self.command.target_vel_x = float(0)
        self.command.target_vel_y = float(0)
        self.command.target_vel_z = float(0.0)

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        self.command.target_yaw = 0.0 #(1-momentum)*(euler[2] + action[2])+momentum*self.command.target_yaw

        return self.command
    
    def load_rl_policy(self, policy_path):
        print("============ policy_path: ", policy_path)
        policy_dir = policy_path  + "/policy.pth"
        rms_dir = policy_path + "/rms.npz"

        act_mean = np.array([0.0, 0.0])[np.newaxis, :] 
        act_std = np.array([0.6, 0.6])[np.newaxis, :]

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
        self.command.target_pos_x = self.state.pos[0]+self.translation_position[0]
        self.command.target_pos_y = self.state.pos[1]+self.translation_position[1]
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
        
        self.command.target_pos_x = self.state.pos[0]+self.translation_position[0]
        self.command.target_pos_y = self.state.pos[1]+self.translation_position[1]
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

    def LaserScan_to_obs_vec(self, obstacles: LaserScan):
        obstacle_length = len(obstacles.ranges)
        angle_min = obstacles.angle_min
        angle_max = obstacles.angle_max
        rad_list = []
        for theta in self.theta_list[::-1]:
            deg = -theta
            rad_list.append(deg*np.pi/180)
        for theta in self.theta_list:
            deg = theta
            rad_list.append(deg*np.pi/180)
        obs_vec = np.empty(0)
        for rad in rad_list:
            index = int(((rad-angle_min)/(angle_max-angle_min))*obstacle_length)
            length = obstacles.ranges[index]
            if length<=self.max_detection_range:
                length/=self.max_detection_range
            if length>self.max_detection_range:
                #include inf (this means there are no data, but I limit this case is larger than range_max)
                length=1
            length-=self.body_size/self.max_detection_range
            obs_vec = np.append(obs_vec,length)
            # obs_vec = np.append(obs_vec,length) 
        # print("conversion_time: ", finish-start) <0.001s
        return obs_vec
    def calc_tilt(self):
        rotation_matrix = R.from_quat(self.state.att)
        self.yaw: np.float32 = rotation_matrix.as_euler('xyz', degrees=True)[2] # deg
        self.tilt = np.arccos(rotation_matrix.as_matrix()[2,2])

    def bad_collision(self, obs_vec):
        return self.land_tilt < self.tilt and np.min(obs_vec*self.max_detection_range) < self.body_size + self.collision_distance
    
    def landing_position_setting(self, obs_vec):
        # set the opposite direction of the nearest obstacle of the landing position
        dist = self.body_size * (1-np.cos(self.tilt)) + self.dist_backup
        min_index = np.argmin(obs_vec)
        direction = -self.theta_list[-min_index-1] if min_index < self.theta_num else self.theta_list[min_index-self.theta_num] # deg
        self.command.target_pos_x = self.state.pos[0]+self.translation_position[0] - dist*np.cos(np.deg2rad(self.yaw + direction))
        self.command.target_pos_y = self.state.pos[1]+self.translation_position[1] - dist*np.sin(np.deg2rad(self.yaw + direction)) #move to the opposite direction of the nearest obstacle
        self.command.target_pos_z = 1.0
        self.command.target_vel_x = float(0)
        self.command.target_vel_y = float(0)
        self.command.target_vel_z = float(0.0)
        self.command.target_yaw = 0.0
        return self.command

    
    def is_landing(self):
        diff = np.array([self.state.pos[0]+self.translation_position[0] - self.command.target_pos_x,
            self.state.pos[1]+self.translation_position[1] - self.command.target_pos_y])
        return np.linalg.norm(diff) < self.landing_dist_threshold


    def landing(self):
        self.land_pub.publish(Empty())

    def is_halt(self,obs_vec):
        return self.max_halt_tilt < self.tilt and np.min(obs_vec*self.max_detection_range) < self.body_size + self.collision_distance

    def is_force_landing(self):
        diff = np.array([self.state.pos[0]+self.translation_position[0] - self.command.target_pos_x,
            self.state.pos[1]+self.translation_position[1] - self.command.target_pos_y])
        return self.force_landing_dist_threshold < np.linalg.norm(diff)

        

if __name__ == '__main__':
    agile_pilot_node = AgilePilotNode()
    rospy.spin()