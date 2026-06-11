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
set(TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_SOURCES
    ${SHARED_DIR}/apriori_recommender.cpp
    ${SHARED_DIR}/needle_technique_classifier.cpp
    ${SHARED_DIR}/meridian_energy_balancer.cpp
    ${SHARED_DIR}/q_learning_advisor.cpp
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
    ${TEST_INCLUDE}
)

target_link_libraries(tcm_tests PRIVATE Threads::Threads)

if(WIN32)
    target_link_libraries(tcm_tests PRIVATE ws2_32 mswsock)
endif()

add_test(NAME tcm_unit_tests COMMAND tcm_tests)
