#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard
import threading
import sys

class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')
        
        # Declare parameters
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('update_rate', 0.1)
        
        # Get parameters
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # Create publisher for cmd_vel
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Flags for key states
        self.keys_pressed = set()
        
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.keyboard_listener.start()
        
        # Start publisher timer
        self.timer = self.create_timer(self.update_rate, self.publish_velocity)
        
        # Print instructions
        self.print_instructions()

    def print_instructions(self):
        instructions = """
Keyboard Control Instructions:
----------------------------
Arrow Keys or WASD:
↑ or W: Move forward
↓ or S: Move backward
← or A: Rotate left
→ or D: Rotate right

Speed Control:
+ : Increase speed
- : Decrease speed

Q : Quit

Current speeds:
Linear speed: {} m/s
Angular speed: {} rad/s
        """.format(self.linear_speed, self.angular_speed)
        print(instructions)

    def on_press(self, key):
        try:
            # Convert key to string representation
            if isinstance(key, keyboard.KeyCode):
                key_str = key.char
            else:
                key_str = str(key)
            
            # Add key to pressed keys set
            self.keys_pressed.add(key_str)
            
            # Handle speed adjustments
            if key_str == '+':
                self.linear_speed += 0.1
                self.angular_speed += 0.1
                print(f"\rSpeeds - Linear: {self.linear_speed:.1f} m/s, Angular: {self.angular_speed:.1f} rad/s")
            elif key_str == '-':
                self.linear_speed = max(0.1, self.linear_speed - 0.1)
                self.angular_speed = max(0.1, self.angular_speed - 0.1)
                print(f"\rSpeeds - Linear: {self.linear_speed:.1f} m/s, Angular: {self.angular_speed:.1f} rad/s")
            
            # Handle quit
            if key_str.lower() == 'q':
                print("\nExiting...")
                self.stop_robot()
                sys.exit(0)
                
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            # Convert key to string representation
            if isinstance(key, keyboard.KeyCode):
                key_str = key.char
            else:
                key_str = str(key)
            
            # Remove key from pressed keys set
            self.keys_pressed.discard(key_str)
            
        except AttributeError:
            pass

    def update_velocity(self):
        # Reset velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Update based on pressed keys
        for key in self.keys_pressed:
            if key in ['Key.up', 'w', 'W']:
                self.linear_velocity = self.linear_speed
            elif key in ['Key.down', 's', 'S']:
                self.linear_velocity = -self.linear_speed
            elif key in ['Key.left', 'a', 'A']:
                self.angular_velocity = self.angular_speed
            elif key in ['Key.right', 'd', 'D']:
                self.angular_velocity = -self.angular_speed

    def publish_velocity(self):
        # Update velocity based on current key states
        self.update_velocity()
        
        # Create and publish Twist message
        msg = Twist()
        msg.linear.x = float(self.linear_velocity)
        msg.angular.z = float(self.angular_velocity)
        self.publisher.publish(msg)

    def stop_robot(self):
        # Publish zero velocity
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = KeyboardController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        if 'controller' in locals():
            controller.stop_robot()
        rclpy.shutdown()

if __name__ == '__main__':
    main()