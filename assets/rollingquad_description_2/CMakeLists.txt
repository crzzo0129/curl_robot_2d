cmake_minimum_required(VERSION 3.8)
project(rollingquad_description_2)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf meshes mjcf launch rviz
  DESTINATION share/${PROJECT_NAME})
ament_package()
