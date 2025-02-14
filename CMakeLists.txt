cmake_minimum_required(VERSION 3.8)
project(flo_base)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SERIAL libserial REQUIRED)

# Include directories
include_directories(
  include
  ${SERIAL_INCLUDE_DIRS}
)

# Serial port node
add_executable(serial_port src/serial_port.cpp)
target_link_libraries(serial_port
  ${SERIAL_LIBRARIES}
)
ament_target_dependencies(serial_port
  rclcpp
  std_msgs
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  flo_base/base_controller.py
  flo_base/keyboard_controller.py
  flo_base/person_follower.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ executables
install(TARGETS
  serial_port
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
