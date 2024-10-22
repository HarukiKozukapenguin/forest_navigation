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
    speed_up = 0
    test = 1
    move_to_test = 2

class PosNeg:
    pos = 0
    neg = 1

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
        self.exec_max_gain = 0.5
        self.speed_up_distance = 0.5
        self.stop_distance = 0.5
        self.x_range = 2.0
        self.vel_threshold = 0.6
        self.vel_stopped = 0.1
        self.pos_start = 0.1
    
    def initialize_variable(self):
        self.publish_commands = False
        self.pos_nav_mode = False
        self.command = FlightNav()
        self.geo_condition = GeoCondition.speed_up
        self.pos_neg = PosNeg.pos
        self.change_to_pos = True
        self.state = None

    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True
        self.command.pos_xy_nav_mode = 1
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
        # if self.pos_neg == PosNeg.pos:
        x_pos = self.state.pos[0]
        # if self.pos_neg == PosNeg.neg:
        #     x_pos = self.x_range - self.state.pos[0]
        vel_x = self.state.vel[0]

        if x_pos < self.speed_up_distance: 
            self.geo_condition = GeoCondition.speed_up
        elif x_pos < self.x_range - self.speed_up_distance and self.vel_stopped < abs(vel_x):
            self.geo_condition = GeoCondition.test
        else:
            self.geo_condition = GeoCondition.move_to_test

        if self.publish_commands:
            self.linvel_pub.publish(self.command)

class PosSpeedUp(smach.State):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['PosTest','exit'])
        self.counter = 0
        self.inverval_num = rospy.get_param("~inverval_num")
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state PosSpeedup')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        if self.counter < self.inverval_num:
            self.counter += 1
            print("counter: ", self.counter)
            while acc_check_node.geo_condition == GeoCondition.speed_up:
                self.vel_set()
                rospy.sleep(0.01)
            return 'PosTest'
        else:
            self.finish()
            while True:
                rospy.sleep(1)
            return 'exit'

    def vel_set(self):
        self.acc_check_node.command.pos_xy_nav_mode = 1
        self.acc_check_node.command.target_pos_x = self.acc_check_node.state.pos[0]+self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = self.acc_check_node.vel_threshold
        self.acc_check_node.command.target_acc_x = 0.0

    def finish(self):
        self.acc_check_node.command.pos_xy_nav_mode = 2 #position mode
        self.acc_check_node.command.target_pos_x = self.acc_check_node.speed_up_distance  + self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = 0.0

class PosTest(PosSpeedUp):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['PosMoveToInit'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state PosTest')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.test:
            self.act_set()
            rospy.sleep(0.01)
            # rospy.loginfo("vel: %f", self.acc_check_node.state.vel[0])
            # rospy.loginfo("vel_threshold: %f", self.acc_check_node.vel_threshold)
        return 'PosMoveToInit'

    def act_set(self):
        self.acc_check_node.command.pos_xy_nav_mode = 3
        self.acc_check_node.command.target_pos_x = self.acc_check_node.state.pos[0]+self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = self.acc_check_node.state.vel[0]
        action = np.array([self.acc_check_node.exec_max_gain, 0.0])
        self.acc_check_node.command.target_acc_x = action[0]

class PosMoveToInit(PosSpeedUp):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['PosSpeedUp'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state PosMoveToInit')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.move_to_test and acc_check_node.pos_start < abs(self.acc_check_node.state.pos[0] - self.acc_check_node.x_range):
            self.pos_set()
            rospy.sleep(0.01)
            # rospy.loginfo("vel: %f", self.acc_check_node.state.vel[0])
            # rospy.loginfo("vel_threshold: %f", self.acc_check_node.vel_threshold)
            # rospy.loginfo("self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold: %d", self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold)
        if self.acc_check_node.geo_condition == GeoCondition.speed_up:
            self.acc_check_node.pos_neg = PosNeg.neg
            self.geo_condition = GeoCondition.speed_up
        return 'PosSpeedUp'

    def pos_set(self):
        self.acc_check_node.command.pos_xy_nav_mode = 2
        self.acc_check_node.command.target_pos_x = self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = 0.0
        self.acc_check_node.command.target_acc_x = 0.0


class NegSpeedUp(PosSpeedUp):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['NegTest'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state NegSpeedUp')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.speed_up:
            self.vel_set()
            rospy.sleep(0.01)
        return 'NegTest'

    def vel_set(self):
        self.acc_check_node.command.pos_xy_nav_mode = 1
        self.acc_check_node.command.target_pos_x = self.acc_check_node.state.pos[0]+self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = -self.acc_check_node.vel_threshold
        self.acc_check_node.command.target_vel_x = 0.0

class NegTest(PosTest):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['NegMoveToInit'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state NegTest')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.test:
            self.act_set()
            rospy.sleep(0.01)
            # rospy.loginfo("vel: %f", self.acc_check_node.state.vel[0])
            # rospy.loginfo("vel_threshold: %f", self.acc_check_node.vel_threshold)
        return 'NegMoveToInit'

    def act_set(self):
        self.acc_check_node.command.pos_xy_nav_mode = 3
        self.acc_check_node.command.target_pos_x = self.acc_check_node.state.pos[0]+self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = self.acc_check_node.state.vel[0]
        action = np.array([-self.acc_check_node.exec_max_gain, 0.0])
        self.acc_check_node.command.target_acc_x = action[0]

class NegMoveToInit(PosMoveToInit):
    def __init__(self, acc_check_node):
        smach.State.__init__(self, outcomes=['PosSpeedUp'])
        self.acc_check_node = acc_check_node

    def execute(self, userdata):
        rospy.loginfo('Executing state PosMoveToInit')
        while self.acc_check_node.state is None:
            rospy.sleep(0.01)
        while acc_check_node.geo_condition == GeoCondition.move_to_test and acc_check_node.pos_start < abs(self.acc_check_node.state.pos[0]):
            self.pos_set()
            rospy.sleep(0.01)
            # rospy.loginfo("vel: %f", self.acc_check_node.state.vel[0])
            # rospy.loginfo("vel_threshold: %f", self.acc_check_node.vel_threshold)
            # rospy.loginfo("self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold: %d", self.acc_check_node.state.vel[0] < acc_check_node.vel_threshold)
        if self.acc_check_node.geo_condition == GeoCondition.speed_up:
            self.acc_check_node.pos_neg = PosNeg.pos
            self.geo_condition = GeoCondition.speed_up
        return 'PosSpeedUp'

    def pos_set(self):
        self.acc_check_node.command.pos_xy_nav_mode = 2
        self.acc_check_node.command.target_pos_x = self.acc_check_node.translation_position[0]
        self.acc_check_node.command.target_vel_x = 0.0
        self.acc_check_node.command.target_acc_x = 0.0

if __name__ == '__main__':
    acc_check_node = AccCheck()
    sm = smach.StateMachine(outcomes=['exit'])
    with sm:
        smach.StateMachine.add('POSSPEEDUP', PosSpeedUp(acc_check_node),
                               transitions={'PosTest':'POSTEST'})
        smach.StateMachine.add('POSTEST', PosTest(acc_check_node),
                               transitions={'PosMoveToInit':'POSMOVETOINIT'})
        smach.StateMachine.add('POSMOVETOINIT', PosMoveToInit(acc_check_node),
                               transitions={'PosSpeedUp':'POSSPEEDUP'})
        # smach.StateMachine.add('NEGSPEEDUP', NegSpeedUp(acc_check_node),
        #                        transitions={'NegTest':'NEGTEST'})
        # smach.StateMachine.add('NEGTEST', NegTest(acc_check_node),
        #                        transitions={'NegMoveToInit':'NEGMOVETOINIT'})
        # smach.StateMachine.add('NEGMOVETOINIT', NegMoveToInit(acc_check_node),
        #                        transitions={'PosSpeedUp':'POSSPEEDUP'})

    sis = smach_ros.IntrospectionServer('acc_check_server', sm, '/ACC_CHECK')
    sis.start()

    outcome = sm.execute()
