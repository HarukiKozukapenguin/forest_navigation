#!/usr/bin/env python3
print("initialize acc_check")
import rospy

from aerial_robot_msgs.msg import FlightNav
import numpy as np
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from aerial_robot_msgs.msg import FlightNav
from scipy.spatial.transform import Rotation as R 

class AgileQuadState:
    def __init__(self, quad_state,transition):
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
        

class AccCheck:
    def __init__(self) -> None:
        rospy.init_node('acc_check', anonymous=False)
        x = rospy.get_param("~shift_x")
        y = rospy.get_param("~shift_y")
        self.translation_position = np.array([x, y],dtype="float32")
        quad_name = rospy.get_param("~robot_ns")

        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                    queue_size=1, tcp_nodelay=True)
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/uav/cog/odom", Odometry, self.state_callback,
                                    queue_size=1, tcp_nodelay=True)
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/uav/nav", FlightNav,
                                          queue_size=1)
        self.set_variable()
        self.initialize_variable()
    
    def set_variable(self):
        self.exec_max_gain = 3.0
        self.check_distance = 1.5
        self.stop_distance = .5
    
    def initialize_variable(self):
        self.publish_commands = False
        self.command = FlightNav()
        self.command.target = 1
        self.command.pos_xy_nav_mode = 4
        self.command.pos_z_nav_mode = 4

    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True
        self.command.pos_xy_nav_mode = 3
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
    
    def state_callback(self, state_data):
        self.command.header.stamp = state_data.header.stamp
        self.state = AgileQuadState(state_data, self.translation_position)
        if self.state.pos[0]<self.check_distance:
            action = np.array([self.exec_max_gain, 0.0])
            self.command.target_acc_x = action[0]
            self.command.target_acc_y = action[1]
        else:
            self.command.pos_xy_nav_mode = 4
            self.command.target_pos_x = self.check_distance+self.stop_distance + self.translation_position[0]
            self.command.target_pos_y = 0+self.translation_position[1]
            self.command.target_vel_x = 0
            self.command.target_vel_y = 0
        if self.publish_commands:
            self.linvel_pub.publish(self.command)
    

if __name__ == '__main__':
    acc_check_node = AccCheck()
    rospy.spin()