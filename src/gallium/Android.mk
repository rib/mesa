# Mesa 3-D graphics library
#
# Copyright (C) 2010-2011 Chia-I Wu <olvaffe@gmail.com>
# Copyright (C) 2010-2011 LunarG Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# src/gallium/Android.mk

GALLIUM_TOP := $(call my-dir)
GALLIUM_COMMON_MK := $(GALLIUM_TOP)/Android.common.mk

SUBDIRS := auxiliary
SUBDIRS += auxiliary/pipe-loader

#
# Gallium drivers and their respective winsys
#

# swrast
ifneq ($(filter swrast,$(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/sw/dri drivers/softpipe
endif

# freedreno
ifneq ($(filter freedreno, $(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/freedreno/drm drivers/freedreno
endif

# i915g
ifneq ($(filter i915g, $(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/i915/drm drivers/i915
endif

# nouveau
ifneq ($(filter nouveau, $(MESA_GPU_DRIVERS)),)
SUBDIRS += \
	winsys/nouveau/drm \
	drivers/nouveau
endif

# r300g/r600g/radeonsi
ifneq ($(filter r300g r600g radeonsi, $(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/radeon/drm
ifneq ($(filter r300g, $(MESA_GPU_DRIVERS)),)
SUBDIRS += drivers/r300
endif
ifneq ($(filter r600g radeonsi, $(MESA_GPU_DRIVERS)),)
SUBDIRS += drivers/radeon
ifneq ($(filter r600g, $(MESA_GPU_DRIVERS)),)
SUBDIRS += drivers/r600
endif
ifneq ($(filter radeonsi, $(MESA_GPU_DRIVERS)),)
SUBDIRS += drivers/radeonsi
SUBDIRS += winsys/amdgpu/drm
endif
endif
endif

# vc4
ifneq ($(filter vc4, $(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/vc4/drm drivers/vc4
endif

# virgl
ifneq ($(filter virgl, $(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/virgl/drm winsys/virgl/vtest drivers/virgl
endif

# vmwgfx
ifneq ($(filter vmwgfx, $(MESA_GPU_DRIVERS)),)
SUBDIRS += winsys/svga/drm drivers/svga
endif

# Gallium state trackers and target for dri
SUBDIRS += state_trackers/dri targets/dri

include $(call all-named-subdir-makefiles,$(SUBDIRS))
