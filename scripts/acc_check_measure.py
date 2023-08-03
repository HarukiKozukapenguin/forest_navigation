#!/usr/bin/env python3
print("initialize acc_check_measure")
import rospy

import numpy as np
from nav_msgs.msg import Odometry
from aerial_robot_msgs.msg import FlightNav
from spinal.msg import FourAxisCommand
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Quaternion
import collections
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R
import smach
import smach_ros
from typing import Union

class AccCheckMeasure:
    def __init__(self) -> None:
        rospy.init_node('acc_check_measure')
        quad_name = rospy.get_param("~robot_ns")
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/uav/cog/odom", Odometry, self.state_callback,
                                    queue_size=1, tcp_nodelay=True)
        self.nav_cmd_sub = rospy.Subscriber("/" + quad_name + "/uav/nav", FlightNav, self.cmd_callback,
                            queue_size=1, tcp_nodelay=True)
        self.att_cmd_sub = rospy.SubscribeListener("/" + quad_name + "/uav/attitude_cmd", FourAxisCommand , self.att_cmd_callback,
                            queue_size=1, tcp_nodelay=True)
        self.delay_result_pub = rospy.Publisher("/" + quad_name + "debug/time_delay_result", Float32MultiArray,
                                    queue_size=1)
        self.set_variable()
        self.initialize_variable()
        
    def set_variable(self):
        self.cmd_diff = 0.1
        self.pitch_deg_diff = 0.1
        self.before_data_collection_num:int = 1
        self.after_data_collection_num:int = 4
        self.show_result_time:float = 10.0

    def initialize_variable(self):
        self.past_cmd = None
        self.switch_time = None
        self.past_deg_data = collections.deque([], self.before_data_collection_num)
        self.after_deg_data = collections.deque([], self.after_data_collection_num)
        self.model = None
        self.tau_list = []
        self.act_delay_list = []
        self.stable_deg = None
        self.current_nav_cmd : Union[float, None] = None

    def cmd_callback(self, cmd_data):
        current_cmd = cmd_data.target_acc_x
        if self.past_cmd is not None and abs(current_cmd - self.past_cmd) > self.cmd_diff:
            self.change_cmd = True
            print("change_cmd")
            self.switch_time_toSec = cmd_data.header.stamp.to_sec()
            self.switch_time = cmd_data.header.stamp
        self.past_cmd = current_cmd
        self.cmd_time = cmd_data.header.stamp
        if self.cmd_time - self.switch_time > rospy.Duration(self.show_result_time):
            self.show_result()
    
    def state_callback(self, state_data):
        ori: Quaternion = state_data.pose.pose.orientation
        att_quaternion: np.array = np.array([ori.x, ori.y, ori.z, ori.w])
        r = R.from_quat(att_quaternion)
        pitch_deg: np.float32 = r.as_euler('xyz', degrees=True)[1] # deg
        if self.stable_deg == None:
            self.stable_deg = pitch_deg
        if self.change_cmd:
            self.change_cmd = False
            self.stable_deg = pitch_deg
            if self.model is not None:
                tau_time = self.model.predict(self.stable_deg)
                tau = tau_time - self.change_time
                self.tau_list.append(tau)
                print("tau: {}".format(tau))

        state_time = state_data.header.stamp.to_sec()
        if abs(pitch_deg - self.stable_deg) < self.pitch_deg_diff:
            self.past_deg_data.append([pitch_deg, state_time])
        else:
            self.after_deg_data.append([pitch_deg, state_time])
            self.change_pitch = False
            if (not self.calc_model) and len(self.after_data_collection_num) == self.after_data_collection_num:
                self.calc_model = True
                self.model = self.predict_time(self.past_deg_data, self.after_deg_data)
                self.change_time = self.model.predict(self.stable_deg)
                act_delay = self.change_time - self.switch_time_toSec
                self.act_delay_list.append(act_delay)
                print("act_delay: {}".format(act_delay))
    
    def predict_time(self, past_deg_data, after_deg_data) -> LinearRegression:
        past_deg_data = np.array(past_deg_data)
        after_deg_data = np.array(after_deg_data)
        past_deg = past_deg_data[:, 0]
        past_time = past_deg_data[:, 1]
        after_deg = after_deg_data[:, 0]
        after_time = after_deg_data[:, 1]
        deg_data = np.concatenate([past_deg, after_deg])
        time_data = np.concatenate([past_time, after_time])
        model = LinearRegression()
        model.fit(deg_data, time_data)
        return model
    
    def show_result(self):
        delay_result = Float32MultiArray()
        delay_result.data = [np.mean(self.act_delay_list), np.var(self.act_delay_list), \
            np.mean(self.tau_list), np.var(self.tau_list)]
        print("act_delay_mean: {}, act_delay_var: {}, tau_mean: {}, tau_var: {}".\
            format(delay_result.data[0], delay_result.data[1], delay_result.data[2], delay_result.data[3]))
        self.delay_result_pub.publish(delay_result)

# Guideline:
# acc_check_node record data from topic
# state machine check acc_check_node's data and decide to move to next state

class WaitNextNav(smach.State):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['wait_des_att','exit'])
        self.acc_check_node = acc_check_node
        self.current_nav_cmd = None

    def execute(self, userdata):
        rospy.loginfo('Executing state WaitNextNav')
        while True:
            if self.current_nav_cmd is None:
                self.current_nav_cmd = self.acc_check_node.current_nav_cmd
                continue
            elif (self.current_time - self.acc_check_node.switch_state_time) > self.wait_next_nav_time:
                return 'exit'
            elif abs(self.acc_check_node.current_nav_cmd - self.current_nav_cmd) > self.nav_cmd_diff:
                self.current_nav_cmd = self.acc_check_node.current_nav_cmd
                self.acc_check_node.
                return 'wait_des_att'
            rospy.sleep(0.01)

class WaitDesAtt(smach.State):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['watch_response'])
        self.acc_check_node = acc_check_node
        self.current_att_cmd = None

    def execute(self, userdata):
        rospy.loginfo('Executing state WaitDesAtt')
        while True:
            if self.current_att_cmd is None:
                self.current_att_cmd = self.acc_check_node.current_att_cmd
                continue
            elif abs(self.acc_check_node.current_att_cmd - self.current_att_cmd) > self.att_cmd_diff:
                self.current_att_cmd = self.acc_check_node.current_att_cmd
                return 'watch_response'
            rospy.sleep(0.01)

class WatchResponse(smach.State):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['wait_next_nav'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state WatchResponse')
        while True:
            self.check_current_pitch_deg()
            if self.over_10percent_step:
                self.over_10percent_time = self.acc_check_node.current_time
                self.sys_delay = self.acc_check_node.send_nav_cmd_time - self.over_10percent_time
            elif self.over_50percent_step:
                self.delay_time = self.acc_check_node.current_time - self.over_10percent_time
            elif self.over_90percent_step:
                self.rise_time = self.acc_check_node.current_time - self.over_10percent_time
                self.acc_check_node.record_data()
                return 'wait_next_nav'
            rospy.sleep(0.01)

if __name__ == '__main__':
    acc_check_node = AccCheckMeasure()
    sm = smach.StateMachine(outcomes=['exit'])
    with sm:
        smach.StateMachine.add('WaitNextNav', WaitNextNav(acc_check_node),
                        transitions={'wait_des_att':'WaitDesAtt'})
        smach.StateMachine.add('WaitDesAtt', WaitDesAtt(acc_check_node),
                               transitions={'watch_response':'WatchResponse'})
        smach.StateMachine.add('WatchResponse', WatchResponse(acc_check_node),
                               transitions={'wait_next_nav':'WaitNextNav'})

    sis = smach_ros.IntrospectionServer('acc_check_measure_server', sm, '/ACC_MEASURE_CHECK')
    sis.start()

    outcome = sm.execute()