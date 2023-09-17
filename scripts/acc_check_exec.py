#!/usr/bin/env python3
print("initialize acc_check")
import rospy

from aerial_robot_msgs.msg import FlightNav
import numpy as np
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from aerial_robot_msgs.msg import FlightNav
from scipy.spatial.transform import Rotation as R 
import smach
import smach_ros

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

class GeoCondition:
    pos_acc = 0
    buffer_place = 1
    neg_acc = 2

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
        self.exec_max_gain = 0.4
        self.check_distance = 0.5
        self.stop_distance = 0.5
        self.x_range = 2.0
        self.vel_threshold = 0.6
    
    def initialize_variable(self):
        self.publish_commands = False
        self.pos_nav_mode = False
        self.command = FlightNav()
        self.geo_condition = GeoCondition.pos_acc
        self.change_to_pos = True
        self.state = None

    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True
        self.command.pos_xy_nav_mode = 7
        self.command.pos_z_nav_mode = 2
        self.command.control_frame = 0
        self.command.target = 1 #COG
        self.command.target_pos_y = 0.0 + self.translation_position[1]
        self.command.target_pos_z = self.state.pos[2]
        self.command.target_vel_y = 0.0
        self.command.target_vel_z = 0.0
        self.command.target_acc_y = 0.0

        # set yaw cmd from state based (in learning, controller is set by diff of yaw angle)
        rotation_matrix = R.from_quat(self.state.att)
        euler = rotation_matrix.as_euler('xyz')
        self.command.target_yaw = euler[2]
    
    def state_callback(self, state_data):
        self.command.header.stamp = state_data.header.stamp
        self.state = AgileQuadState(state_data, self.translation_position)
        if self.state.pos[0] < self.check_distance:
            self.geo_condition = GeoCondition.pos_acc
        elif self.state.pos[0] < self.x_range - self.check_distance:
            self.geo_condition = GeoCondition.buffer_place
        else:
            self.geo_condition = GeoCondition.neg_acc

        if self.publish_commands:
            self.linvel_pub.publish(self.command)

class PosAcc(smach.State):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['neg_acc_prep','exit'])
        self.counter = 0
        self.inverval_num = rospy.get_param("~inverval_num")
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state PosAcc')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        if self.counter < self.inverval_num:
            self.counter += 1
            print("counter: ", self.counter)
            while acc_check_node.geo_condition == GeoCondition.pos_acc:
                self.act_set()
                rospy.sleep(0.01)
            return 'neg_acc_prep'
        else:
            self.finish()
            while True:
                rospy.sleep(1)
            return 'exit'

    def act_set(self):
        self.acc_check_node.command.target_pos_x = self.acc_check_node.state.pos[0]+self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = self.acc_check_node.state.vel[0]
        action = np.array([self.acc_check_node.exec_max_gain, 0.0])
        self.acc_check_node.command.target_acc_x = action[0]
    
    def finish(self):
        self.acc_check_node.command.pos_xy_nav_mode = 2 #position mode
        self.acc_check_node.command.target_pos_x = self.acc_check_node.check_distance + self.acc_check_node.stop_distance + self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = 0.0

class PosAccPrep(PosAcc):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['pos_acc', 'neg_acc_prep'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state PosAccPrep')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.buffer_place and self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold:
            self.act_set()
            rospy.sleep(0.01)
            rospy.loginfo("vel: %f", self.acc_check_node.state.vel[0])
            rospy.loginfo("vel_threshold: %f", self.acc_check_node.vel_threshold)
            rospy.loginfo("self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold: %d", self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold)
        if acc_check_node.geo_condition == GeoCondition.pos_acc:
            return 'pos_acc'
        else:
            return 'neg_acc_prep'

class NegAcc(smach.State):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['pos_acc_prep'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state NegAcc')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.neg_acc:
            self.act_set()
            rospy.sleep(0.01)
        return 'pos_acc_prep'

    def act_set(self):
        self.acc_check_node.command.target_pos_x = self.acc_check_node.state.pos[0] + self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = self.acc_check_node.state.vel[0]
        action = np.array([-self.acc_check_node.exec_max_gain, 0.0])
        self.acc_check_node.command.target_acc_x = action[0]

class NegAccPrep(NegAcc):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['neg_acc', 'pos_acc_prep'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state NegAccPrep')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.buffer_place and -self.acc_check_node.vel_threshold < self.acc_check_node.state.vel[0]:
            self.act_set()
            rospy.sleep(0.01)
            rospy.loginfo("vel: %f", self.acc_check_node.state.vel[0])
            rospy.loginfo("-vel_threshold: %f", -self.acc_check_node.vel_threshold)
            rospy.loginfo("-self.acc_check_node.state.vel[0] < -acc_check_node.vel_threshold: %d", -self.acc_check_node.vel_threshold < self.acc_check_node.state.vel[0])
        if acc_check_node.geo_condition == GeoCondition.neg_acc:
            return 'neg_acc'
        else:
            return 'pos_acc_prep'

if __name__ == '__main__':
    acc_check_node = AccCheck()
    sm = smach.StateMachine(outcomes=['exit'])
    with sm:
        smach.StateMachine.add('POSACC', PosAcc(acc_check_node),
                               transitions={'neg_acc_prep':'NEGACCPREP'})
        smach.StateMachine.add('NEGACCPREP', NegAccPrep(acc_check_node),
                               transitions={'neg_acc':'NEGACC', 'pos_acc_prep':'POSACCPREP'})
        smach.StateMachine.add('NEGACC', NegAcc(acc_check_node),
                               transitions={'pos_acc_prep':'POSACCPREP'})
        smach.StateMachine.add('POSACCPREP', PosAccPrep(acc_check_node),
                               transitions={'pos_acc':'POSACC', 'neg_acc_prep':'NEGACCPREP'})

    sis = smach_ros.IntrospectionServer('acc_check_server', sm, '/ACC_CHECK')
    sis.start()

    outcome = sm.execute()