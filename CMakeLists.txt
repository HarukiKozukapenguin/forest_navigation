cmake_minimum_required(VERSION 3.0.2)
project(forest_navigation)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
# setting of environmental variable
find_package(catkin REQUIRED
  COMPONENTS
  aerial_robot_control
  aerial_robot_model
  aerial_robot_msgs
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

file(GLOB SCRIPTS_FILES "scripts/*.py")
catkin_install_python(
  PROGRAMS ${SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

