import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
from tf_transformations import quaternion_from_euler

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, '/rviz/goal_pose', 10)
        self.timer = self.create_timer(15.0, self.publish_goal)  # Publish every second

        self.poses = [
            #(3.0, 0.0, 0.0),  # 3 meters front
            # (-3.0, 0.0, 0.0),  # 3 meters back
            # (0.0, 3.0, 0.0),  # 3 meters left
             (0.0, -3.0, 0.0),  # 3 meters right
            # (3.0, 3.0, 0.0),  # 3 meters front-left
            # (3.0, -3.0, 0.0),  # 3 meters front-right
            # (-3.0, 3.0, 0.0),  # 3 meters back-left
            # (-3.0, -3.0, 0.0),  # 3 meters back-right
        ]
        self.current_pose_index = 0

    def calculate_yaw(self, goal_x, goal_y):
        return math.atan2(goal_y, goal_x)

    def publish_goal(self):
        # Get the current pose from the list
        pose = self.poses[self.current_pose_index]

        # Create a PoseStamped message
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'base_footprint'  # Assuming the robot's base frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()

        # Set the position
        goal_msg.pose.position.x = pose[0]
        goal_msg.pose.position.y = pose[1]
        goal_msg.pose.position.z = pose[2]
        yaw = self.calculate_yaw(pose[0], pose[1])

        # Set the orientation (facing forward)
        quaternion = quaternion_from_euler(0, 0, yaw)  # Roll, pitch, yaw
        goal_msg.pose.orientation.x = quaternion[0]
        goal_msg.pose.orientation.y = quaternion[1]
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]

        # Publish the goal
        self.publisher_.publish(goal_msg)
        self.get_logger().info(f'Published goal: {pose}')

        # Update the pose index for the next publication
        self.current_pose_index = (self.current_pose_index + 1) % len(self.poses)

def main(args=None):
    rclpy.init(args=args)
    goal_publisher = GoalPublisher()
    rclpy.spin(goal_publisher)
    goal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()