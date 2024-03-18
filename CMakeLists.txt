cmake_minimum_required(VERSION 3.0.2)
project(reorient_planning)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    nav_msgs
    geometry_msgs
    message_generation
)
# find_package(Eigen3 REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

# set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

add_message_files(
  FILES
  compt_time.msg
  opti_variable.msg
)

# add_service_files(
#   FILES
#   CommandInt.srv
# )

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
    # DEPENDS Eigen3
    # INCLUDE_DIRS include src
    CATKIN_DEPENDS message_runtime
)
