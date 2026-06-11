cmake_minimum_required(VERSION 3.14)
project(tcm_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(SHARED_INCLUDE
    ${CMAKE_SOURCE_DIR}/backend/include
    ${CMAKE_SOURCE_DIR}/backend/shared/include
)
set(SHARED_DIR ${CMAKE_SOURCE_DIR}/backend/src)
set(MODULES_DIR ${CMAKE_SOURCE_DIR}/backend/modules)
set(TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR})

set(MODULE_COMMON_SRCS
    ${MODULES_DIR}/common/src/thread_pool.cpp
)

set(TEST_SOURCES
    ${MODULES_DIR}/acupuncture_recommend/src/acupuncture_recommender.cpp
    ${MODULES_DIR}/maneuver_cnn/src/maneuver_cnn.cpp
    ${MODULES_DIR}/energy_radar/src/energy_radar.cpp
    ${MODULES_DIR}/rl_adapter/src/rl_adapter.cpp
    ${MODULE_COMMON_SRCS}
    ${SHARED_DIR}/acupoint_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_apriori.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_needle_technique.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_meridian_energy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_q_learning.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(tcm_tests ${TEST_SOURCES})

target_include_directories(tcm_tests PRIVATE
    ${SHARED_INCLUDE}
    ${MODULES_DIR}/common/include
    ${MODULES_DIR}/acupuncture_recommend/include
    ${MODULES_DIR}/maneuver_cnn/include
    ${MODULES_DIR}/energy_radar/include
    ${MODULES_DIR}/rl_adapter/include
    ${TEST_INCLUDE}
)

target_link_libraries(tcm_tests PRIVATE Threads::Threads)

if(WIN32)
    target_link_libraries(tcm_tests PRIVATE ws2_32 mswsock)
endif()

add_test(NAME tcm_unit_tests COMMAND tcm_tests)
