cmake_minimum_required(VERSION 3.0.2)
project(reorient_planning)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    nav_msgs
    geometry_msgs
)
find_package(Eigen3 REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
    DEPENDS Eigen3
    INCLUDE_DIRS include src
)

include_directories( include src
                     ${catkin_INCLUDE_DIRS}
                     ${Eigen3_INCLUDE_DIRS} 
)

# set(src_multi_pid_xy ${src_multi_pid_xy} src/multirotor/pid/pid_xy_controller.cpp)
# set(src_multi_pid_xy ${src_multi_pid_xy} src/multirotor/pid/pid_xy_main.cpp)

# # multirotor controller & simulator
# add_executable( pid_xy_ctrller ${src_multi_pid_xy} )
# target_link_libraries( pid_xy_ctrller Eigen3::Eigen ${catkin_LIBRARIES} ${EIGEN_LIBRARIES})
