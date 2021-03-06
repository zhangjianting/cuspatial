﻿cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUSPATIAL_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

find_library(CUDFTESTUTIL_LIBRARY "libcudftestutil.a"
             HINTS "$ENV{CUDF_ROOT}/cpp/build/tests/")

message(STATUS "CUDFTESTUTIL: CUDFTESTUTIL_LIBRARY set to ${CUDFTESTUTIL_LIBRARY}")
message(STATUS "CUDFTESTUTIL: CUDF_TEST_INCLUDE set to ${CUDF_TEST_INCLUDE}")

add_library(cudftestutil STATIC IMPORTED ${CUDFTESTUTIL_LIBRARY})
if (CUDF_TEST_INCLUDE AND CUDFTESTUTIL_LIBRARY)
    set_target_properties(cudftestutil PROPERTIES IMPORTED_LOCATION ${CUDFTESTUTIL_LIBRARY})
endif (CUDF_TEST_INCLUDE AND CUDFTESTUTIL_LIBRARY)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cuspatial cudf
                          cudftestutil rmm cudart cuda "${ARROW_LIB}" ${ZLIB_LIBRARIES} NVCategory
                          NVStrings nvrtc gdal geos_c)
    if(USE_NVTX)
        target_link_libraries(${CMAKE_TEST_NAME} ${NVTX_LIBRARY})
    endif(USE_NVTX)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

option(CMAKE_ENABLE_BENCHMARKS "Enable building cuSpatial benchmarks" OFF)

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    if(CMAKE_ENABLE_BENCHMARKS)
        add_executable(${CMAKE_BENCH_NAME}
                    ${CMAKE_BENCH_SRC})
        set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${CMAKE_BENCH_NAME} cuspatial cudf cudftestutil)
        set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
        add_test(NAME ${CMAKE_BENCH_NAME} COMMAND ${CMAKE_BENCH_NAME})
    endif(CMAKE_ENABLE_BENCHMARKS)
endfunction(ConfigureBench)


###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/libcudacxx/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${CUDF_INCLUDE}"
                    "${CUDF_SRC_INCLUDE}"
                    "${CUDF_TEST_INCLUDE}"
                    "${CUB_INCLUDE}"
                    "${SIMT_INCLUDE}"
                    )
 
###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GDAL_LIBRARIES}"
                 "${CONDA_LINK_DIRS}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}"
                 "${CUDF_LIBRARY}"
                 "${CUSPATIAL_LIBRARY}"
                 )

set(P2PNN_JOIN_REFINEMENT_SMALL_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/p2p_nn_refine_test_small.cu")
ConfigureTest(P2PNN_JOIN_REFINEMENT_SMALL_TEST "${P2PNN_JOIN_REFINEMENT_SMALL_SRC}")

set(QUAD_POLYLINE_FILTERING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/quadtree_polyline_test.cu")
ConfigureTest(QUAD_POLYLINE_FILTERING_TEST "${QUAD_POLYLINE_FILTERING_SRC}")

set(QUAD_POLYGON_FILTERING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/quadtree_polygon_test.cu")
ConfigureTest(QUAD_POLYGON_FILTERING_TEST "${QUAD_POLYGON_FILTERING_SRC}")

set(PIP_JOIN_REFINEMENT_SMALL_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/pip_refine_test_small.cu")
ConfigureTest(PIP_JOIN_REFINEMENT_SMALL_TEST "${PIP_JOIN_REFINEMENT_SMALL_SRC}")

set(PIP_JOIN_REFINEMENT_LARGE_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/pip_refine_test_large.cu")
ConfigureTest(PIP_JOIN_REFINEMENT_LARGE_TEST "${PIP_JOIN_REFINEMENT_LARGE_SRC}")

set(PIP_JOIN_NYCTAXI_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_nyctaxi_test.cu"  
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cpp")
ConfigureTest(PIP_JOIN_NYCTAXI_TEST "${PIP_JOIN_NYCTAXI_TEST_SRC}")

set(PIP_JOIN_NYCTAXI_VERIFY_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_nyctaxi_verify.cu"  
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cu")  
ConfigureTest(PIP_JOIN_NYCTAXI_VERIFY "${PIP_JOIN_NYCTAXI_VERIFY_SRC}")

set(POLYGON_BBOX_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/polygon_bbox_test.cu")
ConfigureTest(POLYGON_BBOX_TEST "${POLYGON_BBOX_TEST_SRC}")

set(POLYLINE_BBOX_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/polyline_bbox_test.cu")
ConfigureTest(POLYLINE_BBOX_TEST "${POLYLINE_BBOX_TEST_SRC}")

set(QUADTREE_POINT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/indexing/quadtree_point_test.cu")
ConfigureTest(QUADTREE_POINT_TEST "${QUADTREE_POINT_TEST_SRC}")

set(COORDINATE_TRANSFORM_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/coordinate_transform_toy.cu")
ConfigureTest(COORDINATE_TRANSFORM_TEST "${COORDINATE_TRANSFORM_TEST_SRC}")

set(HAVERSINE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/haversine_toy.cu")
ConfigureTest(HAVERSINE_TEST "${HAVERSINE_TEST_SRC}")

set(HAUSDORFF_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/hausdorff_toy.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/hausdorff_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/hausdorff_compare.cu")
ConfigureTest(HAUSDORFF_TEST "${HAUSDORFF_TEST_SRC}")

set(POINT_IN_POLYGON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/pip_toy.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/pip_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/pip_compare.cu")
ConfigureTest(POINT_IN_POLYGON_TEST "${POINT_IN_POLYGON_TEST_SRC}")

set(PIP_COMPLEX_POLYGON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/pip_complex_polygon.cu")
ConfigureTest(PIP_COMPLEX_POLYGON_TEST "${PIP_COMPLEX_POLYGON_TEST_SRC}")

set(SHAPEFILE_POYGON_READER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/read_shapefile_polygon_test.cu")
ConfigureTest(SHAPEFILE_POLYGON_READER_TEST "${SHAPEFILE_POYGON_READER_TEST_SRC}")

set(SPATIAL_WINDOW_POINT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/query/spatial_window_test_toy.cu")
ConfigureTest(SPATIAL_WINDOW_POINT_TEST "${SPATIAL_WINDOW_POINT_TEST_SRC}")

set(TRAJECTORY_DISTANCE_SPEED_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/test_trajectory_distance_speed.cu")
ConfigureTest(TRAJECTORY_DISTANCE_SPEED_TEST "${TRAJECTORY_DISTANCE_SPEED_TEST_SRC}")

set(TRAJECTORY_DERIVE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/test_trajectory_derive.cu")
ConfigureTest(TRAJECTORY_DERIVE_TEST "${TRAJECTORY_DERIVE_TEST_SRC}")

set(TRAJECTORY_SPATIAL_BOUND_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/trajectory_spatial_bounds_toy.cu")
ConfigureTest(TRAJECTORY_SPATIAL_BOUND_TEST "${TRAJECTORY_SPATIAL_BOUND_SRC}")

set(TRAJECTORY_SUBSET_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/test_trajectory_subset.cu")
ConfigureTest(TRAJECTORY_SUBSET_TEST "${TRAJECTORY_SUBSET_TEST_SRC}")
