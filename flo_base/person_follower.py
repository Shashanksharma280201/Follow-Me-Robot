#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
import message_filters
import torch
import traceback

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')

        self.get_logger().info('Starting person follower initialization...')

        # Declare parameters
        self.declare_parameter('target_distance', 1.5)
        self.declare_parameter('max_linear_speed', 0.36)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('min_distance', 0.3)
        self.declare_parameter('max_distance', 6.0)
        self.declare_parameter('center_threshold', 50)
        
        # Get parameters
        self.target_distance = self.get_parameter('target_distance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.center_threshold = self.get_parameter('center_threshold').value

        # Initialize components
        self.callback_count = 0
        self.last_callback_time = None
        self.image_processing_active = False
        self.bridge = CvBridge()
        
        # Initialize YOLO model
        try:
            self.get_logger().info('Loading YOLO model...')
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            raise

        # Create QoS profile
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, 
            '/cmd_vel',
            10
        )
        self.marker_pub = self.create_publisher(Marker, 'person_marker', 10)
        self.debug_image_pub = self.create_publisher(Image, 'debug_image', 10)

        # Set up subscribers
        try:
            self.left_image_sub = message_filters.Subscriber(
                self, Image, '/left/image_rect', qos_profile=self.sensor_qos)
            self.depth_sub = message_filters.Subscriber(
                self, Image, '/stereo/depth', qos_profile=self.sensor_qos)
            self.stereo_info_sub = message_filters.Subscriber(
                self, CameraInfo, '/stereo/camera_info', qos_profile=self.sensor_qos)

        except Exception as e:
            self.get_logger().error(f'Failed to create subscribers: {str(e)}')
            raise

        # Synchronize messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_image_sub, self.depth_sub, self.stereo_info_sub],
            queue_size=30,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # Add diagnostic timer
        self.diagnostic_timer = self.create_timer(1.0, self.check_callback_health)
        
        self.get_logger().info('Person follower node initialized successfully')

    def check_callback_health(self):
        """Monitor callback health and publish diagnostics."""
        current_time = self.get_clock().now()
        
        if self.last_callback_time is None:
            self.get_logger().warn('No callbacks received yet!')
        else:
            time_diff = (current_time - self.last_callback_time).nanoseconds / 1e9
            self.get_logger().info(f'Time since last callback: {time_diff:.2f} seconds')
            self.get_logger().info(f'Total callbacks received: {self.callback_count}')
            
            if time_diff > 5.0:
                self.get_logger().warn('No recent callbacks received!')
                self.stop_robot()

    def detect_person(self, image):
        """Detect people in the image using YOLOv5."""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(image_rgb)
            
            person_detections = []
            for det in results.xyxy[0]:
                if int(det[5]) == 0:  # Person class
                    person_detections.append({
                        'bbox': det[:4].cpu().numpy(),
                        'conf': det[4].cpu().numpy()
                    })
            
            return person_detections
            
        except Exception as e:
            self.get_logger().error(f'Error in detect_person: {str(e)}')
            return []

    def calculate_3d_position(self, depth_image, detection, camera_info):
        """Calculate 3D position using depth image and camera parameters."""
        try:
            bbox = detection['bbox']
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Validate coordinates
            if not (0 <= center_x < depth_image.shape[1] and 
                   0 <= center_y < depth_image.shape[0]):
                raise ValueError("Invalid center coordinates")
            
            # Get depth at person's position (average over small area)
            depth_window = depth_image[
                max(0, center_y-2):min(depth_image.shape[0], center_y+3),
                max(0, center_x-2):min(depth_image.shape[1], center_x+3)
            ]
            depth = np.median(depth_window[depth_window > 0])
            
            if depth <= 0:
                raise ValueError("Invalid depth value")
            
            # Convert depth to meters (if not already in meters)
            Z = depth / 1000.0  # Convert from mm to meters
            
            # Use the correct camera intrinsics from k matrix
            fx = camera_info.k[0]  # 399.830322265625
            fy = camera_info.k[4]  # 399.830322265625
            cx = camera_info.k[2]  # 311.4825744628906
            cy = camera_info.k[5]  # 205.3360137939453
            
            # Calculate X and Y using the pinhole camera model
            X = (center_x - cx) * Z / fx
            Y = (center_y - cy) * Z / fy
            
            if not (self.min_distance <= Z <= self.max_distance):
                raise ValueError(f"Calculated depth {Z}m is outside valid range")
                
            return np.array([X, Y, Z])
            
        except Exception as e:
            self.get_logger().error(f'Error in calculate_3d_position: {str(e)}')
            self.get_logger().error(f'Camera params - fx: {camera_info.k[0]}, fy: {camera_info.k[4]}, cx: {camera_info.k[2]}, cy: {camera_info.k[5]}')
            return None

    def find_nearest_person(self, detections, depth_image, camera_info):
        """Find the nearest person from all detected people."""
        nearest_person = None
        min_distance = float('inf')
        
        try:
            for detection in detections:
                position = self.calculate_3d_position(
                    depth_image, 
                    detection, 
                    camera_info
                )
                if position is not None:
                    distance = np.linalg.norm(position)
                    if distance < min_distance and self.min_distance <= distance <= self.max_distance:
                        min_distance = distance
                        nearest_person = {
                            'detection': detection,
                            'position': position,
                            'distance': distance
                        }
            
            return nearest_person
            
        except Exception as e:
            self.get_logger().error(f'Error in find_nearest_person: {str(e)}')
            return None

    def calculate_control(self, nearest_person, image_width):
        """Calculate control commands for following and centering."""
        try:
            if nearest_person is None:
                return 0.0, 0.0
                
            # Distance-based linear velocity
            distance = nearest_person['distance']
            distance_error = distance - self.target_distance
            linear_speed = np.clip(
                distance_error, 
                -self.max_linear_speed, 
                self.max_linear_speed
            )
            
            # Add deadband to linear speed
            if abs(linear_speed) < 0.05:
                linear_speed = 0.0
            
            # Calculate centering-based angular velocity
            bbox = nearest_person['detection']['bbox']
            person_center_x = (bbox[0] + bbox[2]) / 2
            center_error = person_center_x - (image_width / 2)
            angular_speed = -(center_error / (image_width / 2)) * self.max_angular_speed
            
            # Add deadband to angular speed
            if abs(angular_speed) < 0.05:
                angular_speed = 0.0
            
            return linear_speed, angular_speed
            
        except Exception as e:
            self.get_logger().error(f'Error in calculate_control: {str(e)}')
            return 0.0, 0.0

    def image_callback(self, image_msg, depth_msg, camera_info_msg):
        """Process images and generate control commands."""
        try:
            # Update diagnostic info
            self.callback_count += 1
            self.last_callback_time = self.get_clock().now()
            self.image_processing_active = True
            
            # Convert ROS messages to OpenCV format
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            if image is None or depth_image is None:
                raise ValueError("Failed to convert ROS images")
            
            # Detect people
            detections = self.detect_person(image)
            
            if not detections:
                self.get_logger().debug('No person detected')
                self.stop_robot()
                return
            
            # Find nearest person
            nearest_person = self.find_nearest_person(
                detections, 
                depth_image, 
                camera_info_msg
            )
            
            if nearest_person is None:
                self.get_logger().debug('No valid person position found')
                self.stop_robot()
                return
            
            # Publish visualization
            self.publish_visualization(nearest_person['position'])
            
            # Calculate and publish control commands
            linear_speed, angular_speed = self.calculate_control(
                nearest_person,
                image.shape[1]
            )
            
            cmd = Twist()
            cmd.linear.x = float(linear_speed)
            cmd.angular.z = float(angular_speed)
            
            self.get_logger().debug(
                f'Publishing cmd_vel - linear: {cmd.linear.x:.2f}, '
                f'angular: {cmd.angular.z:.2f}'
            )
            
            self.cmd_vel_pub.publish(cmd)
            
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')
            self.get_logger().error(traceback.format_exc())
            self.stop_robot()
        finally:
            self.image_processing_active = False

    def publish_visualization(self, position):
        """Publish visualization marker for detected person."""
        try:
            marker = Marker()
            marker.header.frame_id = "oak_right_camera_optical_frame"  # Updated frame ID
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            
            self.marker_pub.publish(marker)
            
        except Exception as e:
            self.get_logger().error(f'Error in publish_visualization: {str(e)}')

    def stop_robot(self):
        """Stop the robot."""
        try:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().debug('Published stop command')
        except Exception as e:
            self.get_logger().error(f'Error in stop_robot: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PersonFollower()
        rclpy.spin(node)
    except Exception as e:
        print(f'Error in main: {str(e)}')
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()