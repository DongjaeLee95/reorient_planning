#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import numpy as np

pos = np.zeros((3,1))
vel = np.zeros((3,1))
quat = np.zeros((4,1))
om = np.zeros((3,1))

def odom_cb(msg: Odometry):

    pos[0] = msg.pose.pose.position.x
    pos[1] = msg.pose.pose.position.y
    pos[2] = msg.pose.pose.position.z

    quat[0] = msg.pose.pose.orientation.w
    quat[1] = msg.pose.pose.orientation.x
    quat[2] = msg.pose.pose.orientation.y
    quat[3] = msg.pose.pose.orientation.z

    vel[0] = msg.twist.twist.linear.x
    vel[1] = msg.twist.twist.linear.y
    vel[2] = msg.twist.twist.linear.z

    om[0] = msg.twist.twist.angular.x
    om[1] = msg.twist.twist.angular.y
    om[2] = msg.twist.twist.angular.z
    print("pos: ", pos.flatten()) 
    print("quat: ", quat.flatten())
    print("vel: ", vel.flatten())
    print("om: ", om.flatten())
    
if __name__ == '__main__':
    
    try:
        # sub =rospy.Subscriber('Local_plan_with_acceleration_input',bjkim_fa_local_planner,callback_bjkim)
        sub =rospy.Subscriber('/odom',Odometry,odom_cb, tcp_nodelay=True)
        rospy.init_node('simple_sub', anonymous=False)
        
        rate = rospy.Rate(100)
        prev = rospy.Time.now().to_sec()
            
        while not rospy.is_shutdown():
            print("i am in while", rospy.Time.now().to_sec()- prev)
            prev = rospy.Time.now().to_sec()
            rate.sleep()

        # rospy.spin()

    except rospy.ROSInterruptException:
        pass
