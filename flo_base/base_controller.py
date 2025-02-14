#!/usr/bin/env python3

import os
import math
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool

class FloBaseDiff(Node):
    def __init__(self):
        super().__init__('flo_base_diff')

        # Create QoS profile for sensor data
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Declare all parameters
        self.declare_and_get_parameters()
        
        # Keep base always enabled
        self.enable = True
        
        # Add heartbeat parameters
        self.declare_parameter('hb_timeout', 1.0)
        self.declare_parameter('hb_hz', 10.0)
        self.hb_timeout = self.get_parameter('hb_timeout').value
        self.hb_hz = self.get_parameter('hb_hz').value
        self.hb_time = None  # Time of last received HB
        
        # Publishers
        self.serial_device_pub = self.create_publisher(String, "serial_port_drive/in", 10)
        self.odom_pub = self.create_publisher(Odometry, "odom/wheel_encoder", 10)
        self.in_motion_pub = self.create_publisher(Bool, 'motion/status', 10)

        # Subscribers
        self.serial_device_sub = self.create_subscription(
            String,
            "serial_port_drive/out",
            self.serial_device_cb,
            self.sensor_qos
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_cb,
            10
        )

        # Initialize state variables
        self.wl = 0.0  # Left wheel angular velocity
        self.wr = 0.0  # Right wheel angular velocity
        self.odom_time = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.cmd_vel_time = None
        self.in_motion = False

        # Timers
        self.odom_timer = self.create_timer(1.0/self.odom_hz, self.odom_cb)
        self.velocity_observer = self.create_timer(
            0.2,
            self.velocity_observer_cb,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        # Add heartbeat timer
        self.hb_timer = self.create_timer(1.0/self.hb_hz, self.hb_cb)
        
        # Add observer timer for monitoring
        self.observer_timer = self.create_timer(1.0, self.observer_cb)
        
        self.get_logger().info('Base controller initialized and enabled')


    def declare_and_get_parameters(self):
        # Declare and get all parameters
        self.declare_parameter('wheel_diameter', 0.165)
        self.declare_parameter('wheel_to_wheel_separation', 0.37)
        self.declare_parameter('wheel_separation_multiplier', 1.0)
        self.declare_parameter('left_wheel_joint_index', 0)
        self.declare_parameter('right_wheel_joint_index', 1)
        self.declare_parameter('cmd_vel_timeout_enabled', True)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('odom_hz', 30.0)
        
        # Get parameters
        self.wheel_diameter = self.get_parameter('wheel_diameter').value
        self.wheel_to_wheel_separation = self.get_parameter('wheel_to_wheel_separation').value
        self.wheel_separation_multiplier = self.get_parameter('wheel_separation_multiplier').value
        self.left_wheel_joint_index = self.get_parameter('left_wheel_joint_index').value
        self.right_wheel_joint_index = self.get_parameter('right_wheel_joint_index').value
        self.cmd_vel_timeout_enabled = self.get_parameter('cmd_vel_timeout_enabled').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.odom_hz = self.get_parameter('odom_hz').value

    def serial_device_write(self, data):
        msg = String()
        msg.data = data
        self.serial_device_pub.publish(msg)
        self.get_logger().info(f'Sent to serial: {data}')

    def serial_device_cb(self, msg):
        self.get_logger().info(f'Received from serial: {msg.data}')
        data_split = msg.data.split(" ")
        
        if data_split[0] == 'HB':  # Heartbeat
            self.hb_time = self.get_clock().now()
            return
            
        if len(data_split) < 2:
            return
            
        if data_split[0] == 'J':  # Joint data
            try:
                index = int(data_split[1])
                velocity = float(data_split[3])
                self.get_logger().info(f'Processing joint data - index: {index}, velocity: {velocity}')
                self.process_joints_state(index, velocity)
            except (ValueError, IndexError) as e:
                self.get_logger().error(f'Error processing joint data: {e}')

    def process_joints_state(self, index, velocity):
        if index == self.left_wheel_joint_index:
            self.wl = velocity
        elif index == self.right_wheel_joint_index:
            self.wr = velocity

    def forward_kinematics(self, wl, wr):
        vl = (self.wheel_diameter/2) * wl
        vr = -(self.wheel_diameter/2) * wr
        
        v = (vr + vl)/2
        w = (vr - vl)/(self.wheel_separation_multiplier * self.wheel_to_wheel_separation)
        return v, w

    def inverse_kinematics(self, v, w):
        wl = (1/(self.wheel_diameter/2)) * (v - (w * \
            self.wheel_separation_multiplier * (self.wheel_to_wheel_separation/2)))
        wr = -(1/(self.wheel_diameter/2)) * (v + (w * \
            self.wheel_separation_multiplier * (self.wheel_to_wheel_separation/2)))
        return wl, wr

    def cmd_vel_cb(self, msg):
        self.get_logger().info(f'Received cmd_vel - linear: {msg.linear.x}, angular: {msg.angular.z}')

        wl, wr = self.inverse_kinematics(msg.linear.x, msg.angular.z)
        self.serial_device_write(f"J {self.left_wheel_joint_index} {wl:.4f}")
        self.serial_device_write(f"J {self.right_wheel_joint_index} {wr:.4f}")
        
        self.v_cmd = msg.linear.x
        self.w_cmd = msg.angular.z
        self.cmd_vel_time = self.get_clock().now()

    def velocity_observer_cb(self):
        msg = Bool()
        msg.data = self.in_motion
        self.in_motion_pub.publish(msg)

    def odom_cb(self):
        v, w = self.forward_kinematics(self.wl, self.wr)
        self.in_motion = abs(v) > 0.0001 or abs(w) > 0.0001  # Lower threshold
        self.get_logger().info(f'v: {v}, w: {w}, in_motion: {self.in_motion}')

        # Check for command timeout
        if self.cmd_vel_timeout_enabled and self.cmd_vel_time is not None:
            dt = (self.get_clock().now() - self.cmd_vel_time).nanoseconds / 1e9
            if dt >= self.cmd_vel_timeout:
                self.get_logger().debug('Command velocity timeout')
                msg = Twist()
                self.cmd_vel_cb(msg)

        # Publish odometry
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_footprint"

        if self.odom_time is None:
            self.odom_time = self.get_clock().now()
        else:
            dt = (self.get_clock().now() - self.odom_time).nanoseconds / 1e9
            self.odom_time = self.get_clock().now()
            
            # Update pose
            self.x += v * math.cos(self.yaw) * dt
            self.y += v * math.sin(self.yaw) * dt
            self.yaw += w * dt

            # Set the pose
            msg.pose.pose.position.x = self.x
            msg.pose.pose.position.y = self.y
            msg.pose.pose.orientation.w = math.cos(self.yaw/2)
            msg.pose.pose.orientation.z = math.sin(self.yaw/2)

        # Set the twist
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w

        self.odom_pub.publish(msg)

    def hb_service_cb(self, request, response):
        if request.data:
            self.hb_timer.reset()
            self.get_logger().info('Heartbeat enabled')
        else:
            self.hb_timer.cancel()
            self.get_logger().info('Heartbeat disabled')
        response.success = True
        return response

    def hb_cb(self):
        self.serial_device_write("HB")
        self.get_logger().debug('Heartbeat sent')

    def observer_cb(self):
        # Check heartbeat status
        if self.hb_time is not None:
            dt = (self.get_clock().now() - self.hb_time).nanoseconds / 1e9
            if dt >= self.hb_timeout:
                self.get_logger().warn(f'No heartbeat received for {dt:.2f} seconds')
        else:
            self.get_logger().warn('No heartbeat received yet')

        # Check command velocity timeout
        if self.cmd_vel_timeout_enabled and self.cmd_vel_time is not None:
            dt = (self.get_clock().now() - self.cmd_vel_time).nanoseconds / 1e9
            if dt >= self.cmd_vel_timeout:
                if self.v_cmd != 0.0 or self.w_cmd != 0.0:
                    self.get_logger().warn('Command velocity timeout, stopping robot')
                    msg = Twist()
                    self.cmd_vel_cb(msg)

    def enable_base_cb(self, req, res):
        if req.enable:
            self.enable = True
        else:
            # 1. Stop the bot
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_cb(msg)
            # 2. set enable
            self.enable = False
        res.success = True
        return res

def main(args=None):
    rclpy.init(args=args)
    
    node = FloBaseDiff()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()