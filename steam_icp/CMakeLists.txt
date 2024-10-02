cmake_minimum_required(VERSION 3.16.3)
project(steam_icp LANGUAGES CXX CUDA)

set(ENABLE_CUDA ON)

# Compile as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Turn on as many warnings as possible by default.
add_compile_options(-march=native -O3 -pthread -Wall -Wextra)

if(ENABLE_CUDA)
  message("Use CUDA for radar processing")
  # CUDA stuff, automatically detect arch
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
  set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo")
  message("CUDA ARCH FLAGS: ${ARCH_FLAGS}")
  message("CUDA FLAGS: ${CUDA_NVCC_FLAGS}")
  message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(tsl-robin-map REQUIRED CONFIG PATHS ${tessil_DIR} NO_DEFAULT_PATH)
find_package(Ceres REQUIRED CONFIG PATHS ${Ceres_DIR} NO_DEFAULT_PATH)
find_package(OpenMP REQUIRED)
find_package(lgmath REQUIRED)
find_package(steam REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

include_directories(
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

link_libraries(
  glog::glog Eigen3::Eigen tsl::robin_map Ceres::ceres OpenMP::OpenMP_CXX
  lgmath steam
)

file(GLOB_RECURSE STEAM_ICP_SRCS src/*.cpp)

# if(ENABLE_CUDA)
#   
# else()
  # add_executable(${PROJECT_NAME} ${STEAM_ICP_SRCS})
# endif()

file(GLOB_RECURSE STEAM_ICP_SRCS_CU src/*.cu)
message($(STEAM_ICP_SRCS_CU))
add_executable(${PROJECT_NAME} ${STEAM_ICP_SRCS} ${STEAM_ICP_SRCS_CU})
ament_target_dependencies(${PROJECT_NAME}
  rclcpp tf2 tf2_ros tf2_eigen nav_msgs
  pcl_conversions pcl_ros OpenCV
)

if(ENABLE_CUDA)
  target_link_libraries(${PROJECT_NAME} "-L/usr/local/lib/libprofiler.so" profiler ${CUDA_LIBRARIES})
  set_target_properties(${PROJECT_NAME} 
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  )
  add_definitions(-DENABLE_CUDA)
endif()
  
add_executable(simulation simulation/sim.cpp)
ament_target_dependencies(simulation
  rclcpp tf2 tf2_ros tf2_eigen nav_msgs
  pcl_conversions pcl_ros OpenCV
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
  ${PROJECT_NAME}
  simulation
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
