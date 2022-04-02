# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(ov_find_tbb)
    if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" AND NOT TBB_FOUND)
        find_package(TBB COMPONENTS tbb tbbmalloc)

        # try to find TBB via custom scripts if have not found by default
        if(NOT TBB_FOUND AND IEDevScripts_DIR)
            # remove invalid TBB_DIR=TBB_DIR-NOTFOUND from cache
            unset(TBB_DIR CACHE)
            unset(TBB_DIR)

            # use our custom scripts for old TBB versions
            # which are exposed via `export TBBROOT=<tbbroot>`
            # see https://github.com/openvinotoolkit/openvino/pull/1288
            find_package(TBB COMPONENTS tbb tbbmalloc
                         PATHS ${IEDevScripts_DIR}
                         NO_CMAKE_FIND_ROOT_PATH
                         NO_DEFAULT_PATH)
        endif()

        # WA for oneTBB: it does not define TBB_IMPORTED_TARGETS
        if(TBB_FOUND AND NOT TBB_IMPORTED_TARGETS)
            foreach(target TBB::tbb TBB::tbbmalloc)
                if(TARGET ${target})
                    list(APPEND TBB_IMPORTED_TARGETS ${target})
                endif()
            endforeach()
        endif()

        # set variables to parent scope to prevent multiple invocations of find_package(TBB)
        # at the same CMakeLists.txt; invocations in different directories are allowed
        set(TBB_FOUND ${TBB_FOUND} PARENT_SCOPE)
        set(TBB_IMPORTED_TARGETS ${TBB_IMPORTED_TARGETS} PARENT_SCOPE)
        set(TBB_VERSION ${TBB_VERSION} PARENT_SCOPE)
        if (NOT TBB_FOUND)
            set(THREADING "SEQ" PARENT_SCOPE)
            message(WARNING "TBB was not found by the configured TBB_DIR/TBBROOT path.\
                             SEQ method will be used.")
        endif ()
    endif()
endmacro()

function(set_ie_threading_interface_for TARGET_NAME)
    # find TBB
    ov_find_tbb()

    get_target_property(target_type ${TARGET_NAME} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        set(LINK_TYPE "INTERFACE")
    elseif(target_type STREQUAL "EXECUTABLE" OR target_type STREQUAL "OBJECT_LIBRARY" OR
           target_type STREQUAL "MODULE_LIBRARY")
        set(LINK_TYPE "PRIVATE")
    elseif(target_type STREQUAL "STATIC_LIBRARY")
        # Affected libraries: inference_engine_s, openvino_gapi_preproc_s
        # they don't have TBB in public headers => PRIVATE
        set(LINK_TYPE "PRIVATE")
    elseif(target_type STREQUAL "SHARED_LIBRARY")
        # Affected libraries: inference_engine only
        # TODO: why TBB propogates its headers to inference_engine?
        set(LINK_TYPE "PRIVATE")
    else()
        message(WARNING "Unknown target type")
    endif()

    function(ie_target_link_libraries TARGET_NAME LINK_TYPE)
        target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${ARGN})

        # include directories as SYSTEM
        foreach(library IN LISTS ARGN)
            if(TARGET ${library})
                get_target_property(include_directories ${library} INTERFACE_INCLUDE_DIRECTORIES)
                if(include_directories)
                    foreach(include_directory IN LISTS include_directories)
                        # cannot include /usr/include headers as SYSTEM
                        if(NOT "${include_directory}" MATCHES "^/usr.*$")
                            target_include_directories(${TARGET_NAME} SYSTEM BEFORE
                                ${LINK_TYPE} $<BUILD_INTERFACE:${include_directory}>)
                        endif()
                    endforeach()
                endif()
            endif()
        endforeach()
    endfunction()

    set(IE_THREAD_DEFINE "IE_THREAD_SEQ")

    if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
        if (TBB_FOUND)
            set(IE_THREAD_DEFINE "IE_THREAD_TBB")
            ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${TBB_IMPORTED_TARGETS})
        else ()
            set(THREADING "SEQ" PARENT_SCOPE)
            message(WARNING "TBB was not found by the configured TBB_DIR path.\
                             SEQ method will be used for ${TARGET_NAME}")
        endif ()
    elseif (THREADING STREQUAL "OMP")
        if (WIN32)
            set(omp_lib_name libiomp5md)
        else ()
            set(omp_lib_name iomp5)
        endif ()

        if (NOT OpenVINO_SOURCE_DIR)
            if (WIN32)
                set(lib_rel_path ${IE_LIB_REL_DIR})
                set(lib_dbg_path ${IE_LIB_DBG_DIR})
            else ()
                set(lib_rel_path ${IE_EXTERNAL_DIR}/omp/lib)
                set(lib_dbg_path ${lib_rel_path})
            endif ()
        else ()
            set(lib_rel_path ${OMP}/lib)
            set(lib_dbg_path ${lib_rel_path})
        endif ()

        if (NOT OMP_LIBRARIES_RELEASE)
            find_library(OMP_LIBRARIES_RELEASE ${omp_lib_name} ${lib_rel_path} NO_DEFAULT_PATH)
            message(STATUS "OMP Release lib: ${OMP_LIBRARIES_RELEASE}")
            if (NOT LINUX)
                find_library(OMP_LIBRARIES_DEBUG ${omp_lib_name} ${lib_dbg_path} NO_DEFAULT_PATH)
                if (OMP_LIBRARIES_DEBUG)
                    message(STATUS "OMP Debug lib: ${OMP_LIBRARIES_DEBUG}")
                else ()
                    message(WARNING "OMP Debug binaries are missed.")
                endif ()
            endif ()
        endif ()

        if (NOT OMP_LIBRARIES_RELEASE)
            message(WARNING "Intel OpenMP not found. Intel OpenMP support will be disabled. ${IE_THREAD_DEFINE} is defined")
            set(THREADING "SEQ" PARENT_SCOPE)
        else ()
            set(IE_THREAD_DEFINE "IE_THREAD_OMP")

            if (WIN32)
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} /openmp)
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} /Qopenmp)
                ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "-nodefaultlib:vcomp")
            else()
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} -fopenmp)
            endif ()

            # Debug binaries are optional.
            if (OMP_LIBRARIES_DEBUG AND NOT LINUX)
                if (WIN32)
                    ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "$<$<CONFIG:DEBUG>:${OMP_LIBRARIES_DEBUG}>;$<$<NOT:$<CONFIG:DEBUG>>:${OMP_LIBRARIES_RELEASE}>")
                else()
                    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
                        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_DEBUG})
                    else()
                        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
                    endif ()
                endif ()
            else ()
                # Link Release library to all configurations.
                ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
            endif ()
        endif ()
    endif ()

    target_compile_definitions(${TARGET_NAME} ${LINK_TYPE} -DIE_THREAD=${IE_THREAD_DEFINE})

    if (NOT THREADING STREQUAL "SEQ")
        find_package(Threads REQUIRED)
        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} Threads::Threads)
    endif()
endfunction(set_ie_threading_interface_for)
