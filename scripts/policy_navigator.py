#!/usr/bin/env python3
import rospy


from aerial_robot_msgs.msg import ObstacleArray, FlightNav
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Empty, Float64MultiArray, Time, Float64, UInt8
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Vector3, Point

import numpy as np
from rospy.numpy_msg import numpy_msg

import torch
print("torch version: {}".format(torch.__version__))
if not torch.cuda.is_available():
    raise ValueError("cuda is not aviable in pytorch")


from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.utils import get_device
from forest_navigation.middle_layer_network import MiddleLayerActorCriticPolicy
# import csv
import sys


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

class Buffer:
    def __init__(self, min, width, dim):
        self.min = min
        self.width = width
        self.sim_dt = 0.025 # 40Hz
        self.max_size : int = int(np.ceil((self.min + self.width)/self.sim_dt))
        self.dim = dim
        self.buffer = [np.zeros(self.dim)] * int(self.max_size)
        self.past_delay = 0

    def reset(self):
        self.buffer = [np.zeros(self.dim)] * int(self.max_size)
        self.past_delay = 0

    def effect_delay(self, new_data: np.array) -> np.array:
        self.buffer.insert(0, new_data)
        time_delay = np.random.uniform(0, self.width) + self.min
        if self.past_delay - self.sim_dt / 4 < time_delay - self.sim_dt:
            time_delay = self.past_delay - self.sim_dt / 4 + self.sim_dt
        time_step = float(time_delay / self.sim_dt)
        time_idx = int(np.floor(time_step))
        time_frac = time_step - time_idx

        out = self.buffer[time_idx] * (1 - time_frac) + self.buffer[time_idx + 1] * time_frac

        if len(self.buffer) > self.max_size:
            self.buffer.pop()
        self.past_delay = time_delay

        return out

class AgilePilotNode:
    def __init__(self):
        rospy.init_node('policy_navigator', anonymous=False)

        self.ppo_path = rospy.get_param("~ppo_path")
        self.get_from_hokuyo = rospy.get_param("~hokuyo")
        # self.real_transition = rospy.get_param("~real_transition")
        # print(self.real_transition)
        self.flight_state = None
        self.publish_commands = False
        self.stop_navigation = False
        self.state = None
        # init setting param setting to avoid error
        self.no_yaw_poll_y = np.array([0,1,0],dtype="float32")
        self.quad_pos = np.array([0,0,0],dtype="float32")
        self.yaw_rad = 0
        self.goal_lin_vel = np.array([5,0,0],dtype="float32")
        self.real_area = rospy.get_param("~real_area")
        if self.real_area:
            self.world_box = np.array([-0.3, 4.2 ,-1.5, 1.5, 0.2, 2.0],dtype="float32")
        else:
            self.world_box = np.array([-0.3, 60 ,-1.5, 1.5, 0.2, 2.0],dtype="float32")
        # should change when changing position
        x = rospy.get_param("~shift_x")
        y = rospy.get_param("~shift_y")
        self.fixed_flight: bool = rospy.get_param("~fixed_flight")
        self.fixed_flight_pos = 4.0
        self.translation_position = np.array([x, y],dtype="float32")
        # last value of theta_list is 134 for the range of the quadrotor
        self.theta_list = np.array([5,15,25,35,45,55, 65, 75, 85, 95, 105, 115, 125]) # deg
        self.acc_theta_list = np.array([1, 4, 7, 10]) # deg
        self.theta_num = len(self.theta_list)
        self.max_detection_range = 10 #max_detection_range when leraning
        # checked several rosbag, and I found that the normal flight rarely exceeds the tilts over 30 degrees
        self.max_halt_tilt = np.deg2rad(60)
        self.land_tilt = np.deg2rad(40)
        self.goal = False
        # should change depending on world flame's origin

        self.is_delay = rospy.get_param("~delay")
        if self.is_delay:
            self.act_buffer = Buffer(0.020, 0.010, 2)
            self.obs_buffer = Buffer(0.020, 0.010, 49)

        quad_name = rospy.get_param("~robot_ns")

        self.command = FlightNav()
        self.command.target = 1
        self.command.pos_xy_nav_mode = 4
        self.command.pos_z_nav_mode = 0

        self.lstm_states = None
        self.body_r = rospy.get_param("~body_r")  #radius of quadrotor(0.25~0.5)
        self.collision_distance = 0.10
        self.dist_backup = 0.10
        self.landing_dist_threshold = 0.05
        self.force_landing_dist_threshold = 0.40
        self.beta = 0.002 # min distance for linearization
        learning_max_gain = 10.0
        self.exec_max_gain = rospy.get_param("~max_gain")
        self.body_move_area_x = 4.00
        self.body_move_area_y = rospy.get_param("~body_move_area_y")
        self.wall_pos_y = 1.75
        self.min_tree_pos = np.array([0.0,-self.body_move_area_y],dtype="float32") + self.translation_position
        self.max_tree_pos = np.array([self.body_move_area_x,+self.body_move_area_y],dtype="float32") + self.translation_position

        self.min_start_obstacles: int = rospy.get_param("~min_start_obstacles")
        self.enough_obstacles = False

        self.att_noise = np.deg2rad(4.0)
        self.omega_noise = 0.5 # rad/s
        self.vel_conversion = np.sqrt(learning_max_gain/self.exec_max_gain)
        # self.vel_conversion = 1.0
        self.time_constant = rospy.get_param("~time_constant")/self.vel_conversion #0.366
        # self.time_constant = rospy.get_param("~time_constant") #0.366
        # self.time_constant = 0.366

        self.n_act = np.zeros(2)
        self.obs_data = None

        # initialize radius_list of the obs
        self.obstacle_pos_list = []
        self.obstacle_pos_list_num:int = 0

        # load policy
        self.rl_policy = self.load_rl_policy(self.ppo_path)

        rospy.loginfo("Initialization completed!")



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
        self.hokuyo_time_pub = rospy.Publisher("/" + quad_name + "/debug/hokuyo_time", Time, queue_size=1)

        self.obs_obstacle_pub = rospy.Publisher("/" + quad_name + "/debug/obs_polar_pixel", ObstacleArray, queue_size=1)
        self.debug_linvel_pub = rospy.Publisher("/" + quad_name + "/debug/uav/nav", FlightNav,
                                          queue_size=1)
        self.debug_min_obs_margin_pub = rospy.Publisher("/" + quad_name + "/debug/min_obs_margin", Float64,
                                          queue_size=1)

        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)
        self.stop_sub = rospy.Subscriber("/" + quad_name + "/stop_navigation", Empty, self.stop_callback,
                                          queue_size=1, tcp_nodelay=True)
        self.flight_state_sub = rospy.Subscriber("/" + quad_name + "/flight_state", UInt8, self.flight_state_callback,
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
        self.debug_min_obs_dist_sub = rospy.Subscriber("/" + quad_name + "/debug/min_obs_distance_with_body", Float64,
                                          self.min_obs_dist_callback, queue_size=1, tcp_nodelay=True)
        # WIP: evaluate in the case of moving obstacle, and estimated from hokuyo_estimation
        if not self.get_from_hokuyo:
            self.check_obs_num_in_range_sub = rospy.Subscriber("/" + quad_name + "/visualization_marker", MarkerArray,
                                                        self.min_obs_in_range_callback, queue_size=1, tcp_nodelay=True)

    # calc listed obstacle num
    def min_obs_in_range_callback(self, markers_msg):
        # self.obstacle_pos_list is aimed for debug
        self.obstacle_pos_list.clear()
        self.obstacle_pos_list_num = 0
        for tree_data in markers_msg.markers:
            if tree_data.ns == "tree_diameter":
                continue
            tree_pos = Point()
            tree_pos = tree_data.pose.position
            if (self.min_tree_pos[0] <tree_pos.x < self.max_tree_pos[0]) and \
               (self.min_tree_pos[1] <tree_pos.y < self.max_tree_pos[1]):
                self.obstacle_pos_list.append(tree_pos)
                self.obstacle_pos_list_num += 1
        if self.min_start_obstacles <= self.obstacle_pos_list_num:
            self.enough_obstacles = True
            # debug: obstacle number and self.enough_obstacles
        rospy.loginfo_throttle(2, "self.obstacle_pos_list_num is %d", self.obstacle_pos_list_num)
        rospy.loginfo_throttle(2,"self.enough_obstacles is %d", self.enough_obstacles)

    def min_obs_dist_callback(self, min_obs_dist):
        min_obs_margin = min_obs_dist.data - self.body_r
        min_obs_margin_msg = Float64()
        min_obs_margin_msg.data = min_obs_margin
        self.debug_min_obs_margin_pub.publish(min_obs_margin_msg)

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data,self.translation_position)
        self.command.header.stamp = state_data.header.stamp
        att_aray = self.state.att
        rotation = R.from_quat(att_aray)

        euler: np.float32 = rotation.as_euler('xyz', degrees=False) # rad
        self.yaw_rad = euler[2]
        no_yaw_rotation = R.from_euler('xyz', [euler[0], euler[1], 0.0], degrees=False)

        self.no_yaw_poll_y = no_yaw_rotation.as_matrix()[1,:]
        self.quad_pos = self.state.pos

        if self.obs_data is None:
            return

        if self.goal:
            return
        if self.get_from_hokuyo:
            hokuyo_time_msg = Time()
            obs_vec, acc_obs_vec = self.LaserScan_to_obs_vec(self.obs_data)
            hokuyo_time_msg.data = self.obs_data.header.stamp
        else:
            # normalized by max_detection_range
            obs_vec: np.array = np.array(self.obs_data.boxel)
            # obs_vec -= self.body_r/self.max_detection_range
            # normalized by max_detectilog_obs_vecon_range
            acc_obs_vec: np.array  = np.array(self.obs_data.acc_boxel)
            # acc_obs_vec -= self.body_r/self.max_detection_range
        if self.state is None:
            return

        if self.flight_state is None:
            rospy.logwarn_throttle(1.0, "cannot get the flight state from robot, skip")
            return

        if self.flight_state != 5:
            # refer to aerial_robot_control/include/aerial_robot_control/flight_navigation.h
            # hover_state is 5
            # TODO: convert to message
            rospy.logwarn_throttle(1.0, "robot does not reach hover state, skip")
            return


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

        obs_polar_pixel = ObstacleArray()
        obs_polar_pixel.boxel = obs_vec
        obs_polar_pixel.acc_boxel = acc_obs_vec
        self.obs_obstacle_pub.publish(obs_polar_pixel)

        vel_msg = self.run_policy(state=self.state, obs_vec=obs_vec, acc_obs_vec=acc_obs_vec, rl_policy=self.rl_policy)
        # publisher for debugging in any situation
        self.debug_linvel_pub.publish(vel_msg)

        if self.publish_commands and self.enough_obstacles:
            # debug to show whith direction quadrotor go in given position
            # print("x_state: ",self.state.pos[0])
            # print("x_direction: ",vel_msg.target_pos_x-(self.state.pos[0]+self.translation_position[0]))
            # print("y_direction: ",vel_msg.target_pos_y-(self.state.pos[1]+self.translation_position[1]))
            self.linvel_pub.publish(vel_msg)
            if self.get_from_hokuyo:
                self.hokuyo_time_pub.publish(hokuyo_time_msg)


    def flight_state_callback(self, msg):
        self.flight_state = msg.data

    def obstacle_callback(self, obs_data):
        self.obs_data = obs_data
        # obstacle conversion depending on the type of sensor
        # range: 0.0~1.0 [m/self.max_detection_range]

    def random_number(self, range):
        return np.random.rand()*range*2-range


    def run_policy(self, state, obs_vec: np.array, acc_obs_vec: np.array, rl_policy=None):

        t = rospy.get_time()

        a = -1/self.beta
        b = 1-np.log(self.beta)
        log_obs_vec = np.where(obs_vec < self.beta, a*obs_vec+b, -np.log(obs_vec))
        acc_distance: np.array = self.calc_dist_to_acc(acc_obs_vec)
        # obs_vec = np.array(obstacles.boxel)
        # Convert state to vector observation
        goal_vel = self.goal_lin_vel
        att_aray = state.att
        r = R.from_quat(att_aray)
        euler = r.as_euler('xyz')
        rotation_matrix = r.as_matrix()
        # add noise to body tilt for real flight
        body_tilt = np.array([rotation_matrix[0,2], rotation_matrix[1,2]]) + np.array([self.random_number(self.att_noise), self.random_number(self.att_noise)])

        # print("state.pos[0]: ", state.pos[0])
        # print("self.world_box[1]-1.2: ", self.world_box[1]-1.2)
        if not self.goal:
            self.goal = self.world_box[1]-0.7<state.pos[0]
        inside_range = True
        for i in range (3):
            inside_range &= self.world_box[i*2]<state.pos[i]<self.world_box[i*2+1]
        if self.goal or not inside_range:
            if self.goal:
                rospy.loginfo("Goal!")
            if not inside_range:
                rospy.logwarn_throttle(1.0, "Out of range!")
                self.force_landing_pub.publish(Empty())
            self.command.pos_xy_nav_mode = 4
            self.command.target_pos_x = self.world_box[1]+self.translation_position[0]-0.7
            self.command.target_pos_y = -0.50-0.25
            self.command.target_pos_z = state.pos[2]

            self.command.target_vel_x = 0
            self.command.target_vel_y = 0
            self.command.target_vel_z = 0

            # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
            self.command.target_yaw = euler[2]
            return self.command

        policy, obs_mean, obs_var, act_mean, act_std = rl_policy

        wall_vec = np.array([(self.wall_pos_y - self.body_r) - state.pos[1], (self.wall_pos_y - self.body_r) + state.pos[1]])

        obs = np.concatenate([
            np.array([self.body_r]), np.array([self.time_constant]), np.array([self.exec_max_gain]), self.n_act.reshape((2)), state.pos[0:2],
            np.array([state.vel[0]*self.vel_conversion]), np.array([state.vel[1]]), body_tilt,
            np.array([state.omega[0]]) + self.random_number(self.omega_noise), np.array([state.omega[1]]) + self.random_number(self.omega_noise),
            np.where(wall_vec < self.beta, a*wall_vec+b, -np.log(wall_vec)),
            log_obs_vec, acc_distance
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
        if self.is_delay:
            obs = self.obs_buffer.effect_delay(obs)

        obs = obs.reshape(-1, obs.shape[0])
        # print("obs: ", obs)
        norm_obs = self.normalize_obs(obs, obs_mean, obs_var)
        #  compute action

        # print("type(norm_obs) is ",type(norm_obs))
        # print("norm_obs is ",norm_obs.shape)
        # now = rospy.Time.now()
        self.n_act, self.lstm_states = policy.predict(norm_obs, state = self.lstm_states, deterministic=True)

        # diff = rospy.Time.now()-now
        # print("policy_execution_time: ", diff.secs, ", ", diff.nsecs)
        if self.is_delay:
            self.n_act = self.act_buffer.effect_delay(self.n_act)

        if self.fixed_flight_pos < obs[0, 2]:
            self.fixed_flight = False

        if not self.fixed_flight:
            # print(self.n_act)
            # print("self.n_act.shape",self.n_act.shape)
            action = (self.n_act * act_std + act_mean)[0, :]

            # cmd freq is same as simulator? cf. in RL dt = 0.02
            momentum = 0.0
            self.command.target_pos_x = (1-momentum)*(state.pos[0] + self.translation_position[0])+momentum*self.command.target_pos_x
            self.command.target_pos_y = (1-momentum)*(state.pos[1] + self.translation_position[1])+momentum*self.command.target_pos_y
            self.command.target_pos_z = 1.0

            self.command.target_vel_x = state.vel[0]
            self.command.target_vel_y = state.vel[1]
            self.command.target_vel_z = float(0.0)

            self.command.target_acc_x = action[0]
            self.command.target_acc_y = action[1]

            self.command.target_yaw = 0.0 #(1-momentum)*(euler[2] + action[2])+momentum*self.command.target_yaw

        else:
            action = np.array([self.exec_max_gain, 0.0])

            momentum = 0.0
            self.command.target_pos_x = (1-momentum)*(state.pos[0] + self.translation_position[0])+momentum*self.command.target_pos_x
            self.command.target_pos_y = (1-momentum)*(state.pos[1] + self.translation_position[1])+momentum*self.command.target_pos_y
            self.command.target_pos_z = 1.0

            self.command.target_vel_x = float(0)
            self.command.target_vel_y = float(0)
            self.command.target_vel_z = float(0.0)

            self.command.target_acc_x = action[0]
            self.command.target_acc_y = action[1]

            self.command.target_yaw = 0.0 #(1-momentum)*(euler[2] + action[2])+momentum*self.command.target_yaw

        #print("action: ", action)
        #print("state.vel[0]*self.vel_conversion: ", state.vel[0]*self.vel_conversion)
        rospy.loginfo_throttle(1, "action: {}, inferece time: {}".format(action, rospy.get_time() - t))

        return self.command

    def load_rl_policy(self, policy_path):
        # # -- load saved varaiables --
        device = get_device("auto")
        policy_dir = policy_path  + "/policy.pth"
        # self.policy_load_setting(policy_dir, device)

        rms_dir = policy_path + "/rms.npz"

        act_mean = np.array([0.0, 0.0])[np.newaxis, :]
        act_std = np.array([self.exec_max_gain, self.exec_max_gain])[np.newaxis, :]

        rms_data = np.load(rms_dir)
        obs_mean = np.mean(rms_data["mean"], axis=0)
        obs_var = np.mean(rms_data["var"], axis=0)

        # Create policy object
        policy = MiddleLayerActorCriticPolicy.load(policy_dir, device = device)

        return policy, obs_mean, obs_var, act_mean, act_std

    def policy_load_setting(self, policy_dir, device):
        saved_variables = torch.load(policy_dir, map_location=device)
        (saved_variables['data'])["shared_lstm"]=True
        (saved_variables['data'])["enable_critic_lstm"]=False
        torch.save(saved_variables, policy_dir)


    def start_callback(self, data):
        rospy.loginfo("Start publishing commands!")
        self.publish_commands = True
        self.command.pos_xy_nav_mode = 3
        self.command.target_pos_x = self.state.pos[0]+self.translation_position[0]
        self.command.target_pos_y = self.state.pos[1]+self.translation_position[1]
        self.command.target_pos_z = self.state.pos[2]

        self.command.target_vel_x = self.state.vel[0]
        self.command.target_vel_y = self.state.vel[1]
        self.command.target_vel_z = self.state.vel[2]

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        r = R.from_quat(self.state.att)
        euler = r.as_euler('xyz')
        self.command.target_yaw = euler[2]

    def stop_callback(self, data):
        print("Stay current position!")
        self.command.pos_xy_nav_mode = 4
        self.command.target_pos_x = self.state.pos[0]+self.translation_position[0]
        self.command.target_pos_y = self.state.pos[1]+self.translation_position[1]
        self.command.target_pos_z = self.state.pos[2]

        self.command.target_vel_x = 0
        self.command.target_vel_y = 0
        self.command.target_vel_z = 0

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        r = R.from_quat(self.state.att)
        euler = r.as_euler('xyz')
        self.command.target_yaw = euler[2]
        self.linvel_pub.publish(self.command)
        self.publish_commands = False


    def normalize_obs(self, obs, obs_mean, obs_var):
        return (obs - obs_mean) / np.sqrt(obs_var + 1e-8)

    def LaserScan_to_obs_vec(self, obstacles: LaserScan):
        vel_calc_boundary = 0.3
        obstacle_length = len(obstacles.ranges)
        angle_min = obstacles.angle_min
        angle_max = obstacles.angle_max

        rad_list = self.full_range_rad_list_maker(self.theta_list)
        acc_rad_list = self.full_range_rad_list_maker(self.acc_theta_list)

        obs_vec = np.empty(0)
        for rad in rad_list:
            depth_rad = rad - self.yaw_rad
            index = int(((depth_rad-angle_min)/(angle_max-angle_min))*obstacle_length)
            if index<0 or index>=obstacle_length:
                print("Error: depth index in out of range: ",index, file=sys.stderr)
                sys.exit(1)
            obs_length = obstacles.ranges[index]
            if obs_length<=self.max_detection_range:
                obs_length/=self.max_detection_range
            if obs_length>self.max_detection_range:
                #include inf (this means there are no data, but I limit this case is larger than range_max)
                obs_length=1
            Cell = np.array([np.cos(rad), np.sin(rad), 0])
            dist_from_wall_p = self.calc_dist_from_wall(+1, Cell, self.no_yaw_poll_y, self.quad_pos)/self.max_detection_range
            dist_from_wall_n = self.calc_dist_from_wall(-1, Cell, self.no_yaw_poll_y, self.quad_pos)/self.max_detection_range
            dist_from_wall = min([dist_from_wall_p, dist_from_wall_n])
            length = min(obs_length, dist_from_wall)
            # length -= self.body_r/self.max_detection_range
            obs_vec = np.append(obs_vec,length)
        acc_obs_vec = np.empty(0)
        for rad in acc_rad_list:
            if np.linalg.norm(np.array([self.state.vel[0], self.state.vel[1]])) < vel_calc_boundary:
                vel_direction = 0
            else:
                vel_direction = np.arctan2(self.state.vel[1], self.state.vel[0])
            depth_rad = rad - self.yaw_rad
            index = int(((depth_rad+vel_direction-angle_min)/(angle_max-angle_min))*obstacle_length)
            if index<0 or index>=obstacle_length:
                print("Error: depth index in out of range: ",index, file=sys.stderr)
                sys.exit(1)
            obs_length = obstacles.ranges[index]
            if obs_length<=self.max_detection_range:
                obs_length/=self.max_detection_range
            if obs_length>self.max_detection_range:
                #include inf (this means there are no data, but I limit this case is larger than range_max)
                obs_length=1
            Cell = np.array([np.cos(rad), np.sin(rad), 0])
            dist_from_wall_p = self.calc_dist_from_wall(+1, Cell, self.no_yaw_poll_y, self.quad_pos)/self.max_detection_range
            dist_from_wall_n = self.calc_dist_from_wall(-1, Cell, self.no_yaw_poll_y, self.quad_pos)/self.max_detection_range
            dist_from_wall = min(dist_from_wall_p, dist_from_wall_n)
            length = min(obs_length, dist_from_wall)
            # length -= self.body_r/self.max_detection_range
            acc_obs_vec = np.append(acc_obs_vec,length)
        return obs_vec, acc_obs_vec

    def full_range_rad_list_maker(self, theta_list):
        rad_list = []
        for theta in theta_list[::-1]:
            deg = -theta
            rad_list.append(deg*np.pi/180)
        for theta in theta_list:
            deg = theta
            rad_list.append(deg*np.pi/180)
        return rad_list

    def calc_tilt(self):
        r = R.from_quat(self.state.att)
        self.yaw_deg: np.float32 = r.as_euler('xyz', degrees=True)[2] # deg
        self.tilt_ang = np.arccos(r.as_matrix()[2,2])

    def bad_collision(self, obs_vec):
        return self.land_tilt < self.tilt_ang and np.min(obs_vec*self.max_detection_range) < self.body_r + self.collision_distance

    def landing_position_setting(self, obs_vec):
        # set the opposite direction of the nearest obstacle of the landing position
        self.command.pos_xy_nav_mode = 4
        dist = self.body_r * (1-np.cos(self.tilt_ang)) + self.dist_backup
        min_index = np.argmin(obs_vec)
        direction = -self.theta_list[-min_index-1] if min_index < self.theta_num else self.theta_list[min_index-self.theta_num] # deg
        self.command.target_pos_x = self.state.pos[0]+self.translation_position[0] - dist*np.cos(np.deg2rad(self.yaw_deg + direction))
        self.command.target_pos_y = self.state.pos[1]+self.translation_position[1] - dist*np.sin(np.deg2rad(self.yaw_deg + direction)) #move to the opposite direction of the nearest obstacle
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
        return self.max_halt_tilt < self.tilt_ang and np.min(obs_vec*self.max_detection_range) < self.collision_distance

    def is_force_landing(self):
        diff = np.array([self.state.pos[0]+self.translation_position[0] - self.command.target_pos_x,
            self.state.pos[1]+self.translation_position[1] - self.command.target_pos_y])
        return self.force_landing_dist_threshold < np.linalg.norm(diff)

    def calc_dist_to_acc(self, obs_vec: np.array) -> np.array:
        svel = np.inner(self.state.vel[0:2], self.state.vel[0:2])*self.vel_conversion**2
        theta_list: list = [-theta for theta in self.acc_theta_list]
        theta_list.reverse()
        theta_list.extend(self.acc_theta_list.tolist())
        act_list = np.empty(0)
        for dist, theta in zip(obs_vec, theta_list):
            theta = np.deg2rad(theta)
            a = -1/(self.beta**2)
            b = 2/self.beta
            if dist < self.beta:
                reciprocal_dist = a*dist+b
            else:
                reciprocal_dist = 1/dist
            gain_normalized_act = 2*np.sin(theta)*svel*reciprocal_dist/((self.max_detection_range)*np.cos(theta)**2)
            act_list = np.append(act_list, gain_normalized_act)
        return act_list

    def calc_dist_from_wall(self, sign: int, Cell: np.array, poll_y: np.array, quad_pos: np.array) -> float:
        y_d = (sign*(self.wall_pos_y-self.body_r) - quad_pos[1])
        cos_theta = np.dot(Cell, poll_y)
        if cos_theta*y_d<=0:
            return self.max_detection_range
        else:
            return y_d/cos_theta

        

if __name__ == '__main__':
    agile_pilot_node = AgilePilotNode()
    rospy.spin()
