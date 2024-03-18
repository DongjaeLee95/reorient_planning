#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

if __name__ == '__main__':
    
    try:
        # global state_
        rospy.init_node('simple_pub', anonymous=False)
        odom_msg = Odometry()

        odom_pub_ = rospy.Publisher('/odom',Odometry, queue_size=10,tcp_nodelay=True)
        
        rate = rospy.Rate(100)
        i = 0
        while not rospy.is_shutdown():
            odom_msg.pose.pose.position.x = i
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.0

            odom_msg.pose.pose.orientation.w = 1.0
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = 0.0

            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0

            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0

            odom_msg.header.stamp = rospy.Time.now()
            odom_pub_.publish(odom_msg)

            i = i+1
            rate.sleep()
        
    except rospy.ROSInterruptException:
        pass
