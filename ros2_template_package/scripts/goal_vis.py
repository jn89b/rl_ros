#!/usr/bin/env python3
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from drone_interfaces.msg import CtlTraj
from ros2_template_package.config import GOAL_STATE

"""
#TODO:
Not a good idea to put it in this package need to put it in the rviz_drone package
but have to grab the random seeded obstacles from the ros_mpc package, future to do
"""

class GoalViz(Node):
    def __init__(self):
        super().__init__("obstacle_visualizer")
        
        self.marker_pub = self.create_publisher(MarkerArray, 
                                                "goal_marker", 
                                                10)
        self.scale:int = 50
        self.goal_state = GOAL_STATE
        self.id = 0
        self.marker_array: MarkerArray = MarkerArray()
        
    def show_goal(self) -> None:
        x = [GOAL_STATE[0]]
        y = [GOAL_STATE[1]]
        radii = [10.0]

        self.marker_array = MarkerArray()
        
        for i, (x_, y_, r) in enumerate(zip(x, y, radii)):

            marker = Marker()
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.pose.position.x = x_/self.scale
            marker.pose.position.y = y_/self.scale
            marker.pose.position.z = 0.0
            
            marker.id = self.id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "/map"
            marker.ns = "obstacles"+str(i)
            marker.lifetime = rclpy.duration.Duration(seconds=1000).to_msg()
            marker.scale.x = 2.0*r/self.scale
            marker.scale.y = 2.0*r/self.scale
            marker.scale.z = 10.0

            self.marker_array.markers.append(marker)
            self.id += 1
        
        # print("publishing markers", self.marker_array.markers)
        self.marker_pub.publish(self.marker_array)
        
def main()->None:
    rclpy.init()
    goal_viz = GoalViz()

    show_once = False
    while rclpy.ok():
        if not show_once:
            goal_viz.show_goal()
            show_once = True
        # goal_viz.show_goal()


    rclpy.shutdown()
    
if __name__=="__main__":
    main()    
            
        