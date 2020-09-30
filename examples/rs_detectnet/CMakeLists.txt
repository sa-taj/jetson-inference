
set(sources ${sources} 
rs_detectnet.cpp
)

find_package(realsense2 REQUIRED)
cuda_add_executable(rs_detectnet ${sources})
target_include_directories(rs_detectnet PRIVATE ${realsense2_INCLUDE_DIRS})
target_link_libraries(rs_detectnet jetson-inference realsense2)
install(TARGETS rs_detectnet DESTINATION bin)
