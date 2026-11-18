cmake_minimum_required(VERSION 3.8)
project(mobile_bot_description)

find_package(ament_cmake REQUIRED)

# A description package has no code: it only installs data (URDF, configs,
# RViz layouts, launch files) into share/ so other packages can find them
# with get_package_share_directory("mobile_bot_description").
install(
  DIRECTORY urdf config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
