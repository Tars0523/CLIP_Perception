cmake_minimum_required(VERSION 3.0.2)
project(clip_)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
)
find_package(OpenCV REQUIRED)
generate_messages(
   DEPENDENCIES
   sensor_msgs
   std_msgs
)
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES clip_
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
  #DEPENDS system_lib
)

catkin_install_python(PROGRAMS
    src/test.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
