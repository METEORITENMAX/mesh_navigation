cmake_minimum_required(VERSION 3.8)
project(rl_mesh_controller)


# set(Torch_DIR "~/py_env/pytorch_env/lib/python3.10/site-packages/torch/share/cmake/Torch/")
# set(CMAKE_PREFIX_PATH "~/py_env/pytorch_env/lib64/python3.10/site-packages/torch/lib/")
# set(Torch_DIR "/home/max/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
# set(CMAKE_PREFIX_PATH "/home/max/.local/lib/python3.10/site-packages/torch")

# Set the path to the TensorFlow library
# set(TENSORFLOW_INCLUDE_DIR "/home/max/Programs/tensorflow")
# set(ABSL_INCLUDE_DIR "/home/max/Programs/abseil-cpp")
# set(TENSORFLOW_LIBRARY "/home/max/Programs/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so.2.19.0")
set(MKL_INCLUDE_DIR "/usr/include/mkl")
set(MKL_LIBRARY "/usr/lib/x86_64-linux-gnu/libmkl_rt.so")
#set(Torch_DIR "/home/max/py_env/pytorch_env/lib/python3.10/site-packages/torch/share/cmake/Torch")
#set(Torch_DIR "/home/max/Programs/libtorch_cpu_with_deps/libtorch/share/cmake/Torch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_LIBRARIES "/home/max/Programs/libtorch_cpu_with_deps/libtorch/lib/libtorch_cpu.so")
#find_package(Torch REQUIRED)
add_library(${PROJECT_NAME} SHARED src/rl_mesh_controller.cpp src/actor_critic_network.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set(CMAKE_PREFIX_PATH "/home/max/Programs/libtorch_cpu_with_deps/libtorch")
include_directories(/home/max/Programs/libtorch_cpu_with_deps/libtorch/include)
include_directories(/home/max/Programs/libtorch_cpu_with_deps/libtorch/include/torch/csrc/api/include)
find_package(ament_cmake_ros REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(mbf_mesh_core REQUIRED)
find_package(mbf_msgs REQUIRED)
find_package(mbf_utility REQUIRED)
find_package(mesh_map REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(LVR2 REQUIRED)

#find_package(class_loader REQUIRED)
# Set MKL include and library directories
# set(MKL_INCLUDE_DIR "/usr/include/mkl")
# set(MKL_LIBRARY "/usr/lib/x86_64-linux-gnu/libmkl_rt.so")

pluginlib_export_plugin_description_file(mbf_mesh_core rl_mesh_controller.xml)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${LVR2_INCLUDE_DIRS}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE "MESH_CONTROLLER_BUILDING_LIBRARY")
ament_target_dependencies(${PROJECT_NAME} example_interfaces mbf_mesh_core mbf_msgs mbf_utility mesh_map rclcpp tf2_geometry_msgs)
target_link_libraries(${PROJECT_NAME}
    ${LVR2_LIBRARIES}
    "${TORCH_LIBRARIES}"
    "/home/max/Programs/libtorch_cpu_with_deps/libtorch/lib/libc10.so"
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
install(DIRECTORY include DESTINATION include)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(example_interfaces mbf_mesh_core mbf_msgs mbf_utility mesh_map rclcpp tf2_geometry_msgs )
ament_package()