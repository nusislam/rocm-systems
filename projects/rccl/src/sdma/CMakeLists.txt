###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# ADD ROCSHMEM TARGET FOR SDMA FILES
###############################################################################
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    anvil.cpp
    two_shot_allreduce.cpp
)
# Anvil uses HSA runtime iteration and KFD (libhsakmt) for SDMA queue setup.
target_link_libraries(${PROJECT_NAME} PRIVATE hsa-runtime64::hsa-runtime64)
find_library(RCCL_HSAKMT_LIB NAMES hsakmt HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64")
if(NOT RCCL_HSAKMT_LIB)
  message(FATAL_ERROR "Anvil SDMA requires libhsakmt under ROCM_PATH (${ROCM_PATH}).")
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${RCCL_HSAKMT_LIB})

