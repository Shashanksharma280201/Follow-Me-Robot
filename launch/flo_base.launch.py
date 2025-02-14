from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('flo_base')
    config_file = os.path.join(pkg_dir, 'config', 'base_params.yaml')
    
    return LaunchDescription([
        # Load parameters
        DeclareLaunchArgument(
            'params_file',
            default_value=config_file,
            description='Path to the ROS2 parameters file to use'
        ),
        
        # Serial port node
        Node(
            package='flo_base',
            executable='serial_port',
            name='serial_port_drive',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
        
        # Base controller node
        Node(
            package='flo_base',
            executable='base_controller.py',
            name='flo_base_diff',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
        
        # Keyboard controller node
        Node(
            package='flo_base',
            executable='keyboard_controller.py',
            name='keyboard_controller',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        )
    ])