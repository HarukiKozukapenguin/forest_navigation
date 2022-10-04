# forest_navigation

Environment setting:
do setting of [jsk_aerial_robot](<https://github.com/jsk-ros-pkg/jsk_aerial_robot>)
and 
```
source path_to_jsk_aerial_robot_ws/devel/setup.bash
```
```
apt-get install ros-noetic-catkin-virtualenv
mkdir ~/policy_ws/src && cd ~/policy_ws/src
git clone https://github.com/HarukiKozukapenguin/forest_navigation.git
catkin build
```

Execute:
```
source ~/policy_ws/devel/setup.bash
roslaunch forest_navigation navigator.launch
```
In another terminal
```
rostopic pub /multirotor/start_navigation std_msgs/Empty "{}" -1
```
