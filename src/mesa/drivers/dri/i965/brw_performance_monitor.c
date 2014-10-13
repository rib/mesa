/*
 * Copyright © 2013 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file brw_performance_monitor.c
 *
 * Implementation of the GL_AMD_performance_monitor extension.
 *
 * On Gen5+ hardware, we have two sources of performance counter data:
 * the Observability Architecture counters (MI_REPORT_PERF_COUNT), and
 * the Pipeline Statistics Registers.  We expose both sets of raw data,
 * as well as some useful processed values.
 *
 * The Observability Architecture (OA) counters for Gen6+ are documented
 * in a separate document from the rest of the PRMs.  It is available at:
 * https://01.org/linuxgraphics/documentation/driver-documentation-prms
 * => 2013 Intel Core Processor Family => Observability Performance Counters
 * (This one volume covers Sandybridge, Ivybridge, Baytrail, and Haswell.)
 *
 * On Ironlake, the OA counters were called "CHAPS" counters.  Sadly, no public
 * documentation exists; our implementation is based on the source code for the
 * intel_perf_counters utility (which is available as part of intel-gpu-tools).
 */

#include <linux/perf_event.h>

#include <limits.h>

#include <asm/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "main/bitset.h"
#include "main/hash.h"
#include "main/macros.h"
#include "main/mtypes.h"
#include "main/performance_monitor.h"

#include "util/ralloc.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "intel_batchbuffer.h"

#define FILE_DEBUG_FLAG DEBUG_PERFMON

/**
 * i965 representation of a performance monitor object.
 */
struct brw_perf_monitor_object
{
   /** The base class. */
   struct gl_perf_monitor_object base;

   /**
    * BO containing OA counter snapshots at monitor Begin/End time.
    */
   drm_intel_bo *oa_bo;

   /**
    * snapshots [bookend_snapshots_begin, bookend_snapshots_end) within
    * perfmon->bookend_bo belong to this monitor and should be accumulated
    * at the end of monitoring.
    */
   int bookend_snapshots_begin;
   int bookend_snapshots_end;

   /**
    * Storage for OA results accumulated so far.
    *
    * An array indexed by the counter ID in the brw_oa_counter_id enum.
    *
    * When we run out of space in bookend_bo, we accumulate the deltas
    * accrued so far and add them to the value stored here.  Then, we
    * can discard bookend_bo.
    */
   uint64_t oa_accumulator[MAX_OA_COUNTERS];

   /** Indicates whether any snapshots have been accumulated yet */
   bool accumulated;

   /**
    * false while in the unresolved_elements list, and set to true when
    * the final, end MI_RPC snapshot has been accumulated.
    */
   bool resolved;

   /**
    * BO containing starting and ending snapshots for any active pipeline
    * statistics counters.
    */
   drm_intel_bo *pipeline_stats_bo;

   /**
    * Storage for final pipeline statistics counter results.
    */
   uint64_t *pipeline_stats_results;
};

/* attr.config */

#define I915_PERF_OA_CTX_ID_MASK	    0xffffffff
#define I915_PERF_OA_SINGLE_CONTEXT_ENABLE  (1ULL << 32)

#define I915_PERF_OA_FORMAT_SHIFT	    33
#define I915_PERF_OA_FORMAT_MASK	    (0x7ULL << 33)
#define I915_PERF_OA_FORMAT_A13_HSW	    (0ULL << 33)
#define I915_PERF_OA_FORMAT_A29_HSW	    (1ULL << 33)
#define I915_PERF_OA_FORMAT_A13_B8_C8_HSW   (2ULL << 33)
#define I915_PERF_OA_FORMAT_A29_B8_C8_HSW   (3ULL << 33)
#define I915_PERF_OA_FORMAT_B4_C8_HSW	    (4ULL << 33)
#define I915_PERF_OA_FORMAT_A45_B8_C8_HSW   (5ULL << 33)
#define I915_PERF_OA_FORMAT_B4_C8_A16_HSW   (6ULL << 33)
#define I915_PERF_OA_FORMAT_C4_B8_HSW	    (7ULL << 33)

#define I915_PERF_OA_TIMER_EXPONENT_SHIFT   36
#define I915_PERF_OA_TIMER_EXPONENT_MASK    (0x3fULL << 36)

/* FIXME: HACK to dig out the context id from the
 * otherwise opaque drm_intel_context struct! */
struct _drm_intel_context {
   unsigned int ctx_id;
};
static int eu_count = 0; /* FIXME - find a better home */

/** Downcasting convenience macro. */
static inline struct brw_perf_monitor_object *
brw_perf_monitor(struct gl_perf_monitor_object *m)
{
   return (struct brw_perf_monitor_object *) m;
}

#define SECOND_SNAPSHOT_OFFSET_IN_BYTES 2048

/* A random value used to ensure we're getting valid snapshots. */
#define REPORT_ID 0xd2e9c607

/******************************************************************************/

#define COUNTER(name)           \
   {                            \
      .Name = name,             \
      .Type = GL_UNSIGNED_INT,  \
      .Minimum = { .u32 =  0 }, \
      .Maximum = { .u32 = ~0 }, \
   }

#define PERCENTAGE(name)         \
   {                             \
      .Name = name,              \
      .Type = GL_PERCENTAGE_AMD, \
      .Minimum = { .f = 0   },   \
      .Maximum = { .f = 100 },   \
   }

#define COUNTER64(name)              \
   {                                 \
      .Name = name,                  \
      .Type = GL_UNSIGNED_INT64_AMD, \
      .Minimum = { .u64 =  0 },      \
      .Maximum = { .u64 = ~0 },      \
   }

#define GROUP(name, max_active, counter_list)  \
   {                                           \
      .Name = name,                            \
      .MaxActiveCounters = max_active,         \
      .Counters = counter_list,                \
      .NumCounters = ARRAY_SIZE(counter_list), \
   }

/** Performance Monitor Group IDs */
enum brw_counter_groups {
   OA_COUNTERS, /* Observability Architecture (MI_REPORT_PERF_COUNT) Counters */
   PIPELINE_STATS_COUNTERS, /* Pipeline Statistics Register Counters */
};

const static struct gl_perf_monitor_counter gen6_statistics_counters[] = {
   COUNTER64("IA_VERTICES_COUNT"),
   COUNTER64("IA_PRIMITIVES_COUNT"),
   COUNTER64("VS_INVOCATION_COUNT"),
   COUNTER64("GS_INVOCATION_COUNT"),
   COUNTER64("GS_PRIMITIVES_COUNT"),
   COUNTER64("CL_INVOCATION_COUNT"),
   COUNTER64("CL_PRIMITIVES_COUNT"),
   COUNTER64("PS_INVOCATION_COUNT"),
   COUNTER64("PS_DEPTH_COUNT"),
   COUNTER64("SO_NUM_PRIMS_WRITTEN"),
   COUNTER64("SO_PRIM_STORAGE_NEEDED"),
};

/** MMIO register addresses for each pipeline statistics counter. */
const static int gen6_statistics_register_addresses[] = {
   IA_VERTICES_COUNT,
   IA_PRIMITIVES_COUNT,
   VS_INVOCATION_COUNT,
   GS_INVOCATION_COUNT,
   GS_PRIMITIVES_COUNT,
   CL_INVOCATION_COUNT,
   CL_PRIMITIVES_COUNT,
   PS_INVOCATION_COUNT,
   PS_DEPTH_COUNT,
   GEN6_SO_NUM_PRIMS_WRITTEN,
   GEN6_SO_PRIM_STORAGE_NEEDED,
};

const static struct gl_perf_monitor_group gen6_groups[] = {
   GROUP("Pipeline Statistics Registers", INT_MAX, gen6_statistics_counters),
};


/**
 * Haswell:
 *  @{
 */
const static struct gl_perf_monitor_counter gen7_normalized_oa_counters[] = {
   PERCENTAGE("Render Engine Busy"),
   PERCENTAGE("EU Active"),
   PERCENTAGE("EU Stalled"),
   PERCENTAGE("VS EU Active"),
   PERCENTAGE("VS EU Stalled"),
   COUNTER64("Average Cycles per VS Thread"),
   COUNTER64("Average Stalled Cycles per VS Thread"),
   PERCENTAGE("HS EU Active"),
   PERCENTAGE("HS EU Stalled"),
   COUNTER64("Average Cycles per HS Thread"),
   COUNTER64("Average Stalled Cycles per HS Thread"),
   PERCENTAGE("DS EU Active"),
   PERCENTAGE("DS EU Stalled"),
   COUNTER64("Average Cycles per DS Thread"),
   COUNTER64("Average Stalled Cycles per DS Thread"),
   PERCENTAGE("CS EU Active"),
   PERCENTAGE("CS EU Stalled"),
   COUNTER64("Average Cycles per CS Thread"),
   COUNTER64("Average Stalled Cycles per CS Thread"),
   PERCENTAGE("GS EU Active"),
   PERCENTAGE("GS EU Stalled"),
   COUNTER64("Average Cycles per GS Thread"),
   COUNTER64("Average Stalled Cycles per GS Thread"),
   PERCENTAGE("PS EU Active"),
   PERCENTAGE("PS EU Stalled"),
   COUNTER64("Average Cycles per PS Thread"),
   COUNTER64("Average Stalled Cycles per PS Thread"),

   /* XXX: Just for debugging.... */
   COUNTER64("GPU Timestamp"),
   COUNTER64("GPU Clock"),
};

enum brw_oa_counter_id gen7_oa_counter_map[] = {
   OA_RENDER_BUSY_PERCENTAGE,
   OA_EU_ACTIVE_PERCENTAGE,
   OA_EU_STALLED_PERCENTAGE,

   OA_VS_EU_ACTIVE_PERCENTAGE,
   OA_VS_EU_STALLED_PERCENTAGE,
   OA_AVERAGE_VS_THREAD_CYCLES,
   OA_AVERAGE_STALLED_VS_THREAD_CYCLES,

   OA_HS_EU_ACTIVE_PERCENTAGE,
   OA_HS_EU_STALLED_PERCENTAGE,
   OA_AVERAGE_HS_THREAD_CYCLES,
   OA_AVERAGE_STALLED_HS_THREAD_CYCLES,

   OA_DS_EU_ACTIVE_PERCENTAGE,
   OA_DS_EU_STALLED_PERCENTAGE,
   OA_AVERAGE_DS_THREAD_CYCLES,
   OA_AVERAGE_STALLED_DS_THREAD_CYCLES,

   OA_CS_EU_ACTIVE_PERCENTAGE,
   OA_CS_EU_STALLED_PERCENTAGE,
   OA_AVERAGE_CS_THREAD_CYCLES,
   OA_AVERAGE_STALLED_CS_THREAD_CYCLES,

   OA_GS_EU_ACTIVE_PERCENTAGE,
   OA_GS_EU_STALLED_PERCENTAGE,
   OA_AVERAGE_GS_THREAD_CYCLES,
   OA_AVERAGE_STALLED_GS_THREAD_CYCLES,

   OA_PS_EU_ACTIVE_PERCENTAGE,
   OA_PS_EU_STALLED_PERCENTAGE,
   OA_AVERAGE_PS_THREAD_CYCLES,
   OA_AVERAGE_STALLED_PS_THREAD_CYCLES,

   OA_GPU_TIMESTAMP,
   OA_GPU_CORE_CLOCK
};

const static struct gl_perf_monitor_counter gen7_statistics_counters[] = {
   COUNTER64("IA_VERTICES_COUNT"),
   COUNTER64("IA_PRIMITIVES_COUNT"),
   COUNTER64("VS_INVOCATION_COUNT"),
   COUNTER64("HS_INVOCATION_COUNT"),
   COUNTER64("DS_INVOCATION_COUNT"),
   COUNTER64("GS_INVOCATION_COUNT"),
   COUNTER64("GS_PRIMITIVES_COUNT"),
   COUNTER64("CL_INVOCATION_COUNT"),
   COUNTER64("CL_PRIMITIVES_COUNT"),
   COUNTER64("PS_INVOCATION_COUNT"),
   COUNTER64("PS_DEPTH_COUNT"),
   COUNTER64("SO_NUM_PRIMS_WRITTEN (Stream 0)"),
   COUNTER64("SO_NUM_PRIMS_WRITTEN (Stream 1)"),
   COUNTER64("SO_NUM_PRIMS_WRITTEN (Stream 2)"),
   COUNTER64("SO_NUM_PRIMS_WRITTEN (Stream 3)"),
   COUNTER64("SO_PRIM_STORAGE_NEEDED (Stream 0)"),
   COUNTER64("SO_PRIM_STORAGE_NEEDED (Stream 1)"),
   COUNTER64("SO_PRIM_STORAGE_NEEDED (Stream 2)"),
   COUNTER64("SO_PRIM_STORAGE_NEEDED (Stream 3)"),
};

/** MMIO register addresses for each pipeline statistics counter. */
const static int gen7_statistics_register_addresses[] = {
   IA_VERTICES_COUNT,
   IA_PRIMITIVES_COUNT,
   VS_INVOCATION_COUNT,
   HS_INVOCATION_COUNT,
   DS_INVOCATION_COUNT,
   GS_INVOCATION_COUNT,
   GS_PRIMITIVES_COUNT,
   CL_INVOCATION_COUNT,
   CL_PRIMITIVES_COUNT,
   PS_INVOCATION_COUNT,
   PS_DEPTH_COUNT,
   GEN7_SO_NUM_PRIMS_WRITTEN(0),
   GEN7_SO_NUM_PRIMS_WRITTEN(1),
   GEN7_SO_NUM_PRIMS_WRITTEN(2),
   GEN7_SO_NUM_PRIMS_WRITTEN(3),
   GEN7_SO_PRIM_STORAGE_NEEDED(0),
   GEN7_SO_PRIM_STORAGE_NEEDED(1),
   GEN7_SO_PRIM_STORAGE_NEEDED(2),
   GEN7_SO_PRIM_STORAGE_NEEDED(3),
};

const static struct gl_perf_monitor_group gen7_groups[] = {
   GROUP("Observability Architecture Counters", INT_MAX, gen7_normalized_oa_counters),
   GROUP("Pipeline Statistics Registers", INT_MAX, gen7_statistics_counters),
};
/** @} */

/******************************************************************************/

static GLboolean brw_is_perf_monitor_result_available(struct gl_context *, struct gl_perf_monitor_object *);

static void
dump_perf_monitor_callback(GLuint name, void *monitor_void, void *brw_void)
{
   struct gl_context *ctx = brw_void;
   struct gl_perf_monitor_object *m = monitor_void;
   struct brw_perf_monitor_object *monitor = monitor_void;

   const char *resolved = monitor->resolved ? "Resolved" : "";

   DBG("%4d  %-7s %-6s %-10s %-11s <%3d, %3d>  %-6s %-9s\n",
       name,
       m->Active ? "Active" : "",
       m->Ended ? "Ended" : "",
       resolved,
       brw_is_perf_monitor_result_available(ctx, m) ? "Available" : "",
       monitor->bookend_snapshots_begin,
       monitor->bookend_snapshots_end,
       monitor->oa_bo ? "OA BO" : "",
       monitor->pipeline_stats_bo ? "Stats BO" : "");
}

void
brw_dump_perf_monitors(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;
   DBG("Monitors: (Open monitors = %d, OA users = %d)\n",
       brw->perfmon.open_oa_monitors, brw->perfmon.oa_users);
   _mesa_HashWalk(ctx->PerfMonitor.Monitors, dump_perf_monitor_callback, brw);
}

/******************************************************************************/

static bool
monitor_needs_statistics_registers(struct brw_context *brw,
                                   struct gl_perf_monitor_object *m)
{
   return brw->gen >= 6 && m->ActiveGroups[PIPELINE_STATS_COUNTERS];
}

/**
 * Take a snapshot of any monitored pipeline statistics counters.
 */
static void
snapshot_statistics_registers(struct brw_context *brw,
                              struct brw_perf_monitor_object *monitor,
                              uint32_t offset_in_bytes)
{
   struct gl_context *ctx = &brw->ctx;
   const int offset = offset_in_bytes / sizeof(uint64_t);
   const int group = PIPELINE_STATS_COUNTERS;
   const int num_counters = ctx->PerfMonitor.Groups[group].NumCounters;

   intel_batchbuffer_emit_mi_flush(brw);

   for (int i = 0; i < num_counters; i++) {
      if (BITSET_TEST(monitor->base.ActiveCounters[group], i)) {
         assert(ctx->PerfMonitor.Groups[group].Counters[i].Type ==
                GL_UNSIGNED_INT64_AMD);

         brw_store_register_mem64(brw, monitor->pipeline_stats_bo,
                                  brw->perfmon.statistics_registers[i],
                                  offset + i);
      }
   }
}

/**
 * Gather results from pipeline_stats_bo, storing the final values.
 *
 * This allows us to free pipeline_stats_bo (which is 4K) in favor of a much
 * smaller array of final results.
 */
static void
gather_statistics_results(struct brw_context *brw,
                          struct brw_perf_monitor_object *monitor)
{
   struct gl_context *ctx = &brw->ctx;
   const int num_counters =
      ctx->PerfMonitor.Groups[PIPELINE_STATS_COUNTERS].NumCounters;

   monitor->pipeline_stats_results = calloc(num_counters, sizeof(uint64_t));
   if (monitor->pipeline_stats_results == NULL) {
      _mesa_error_no_memory(__func__);
      return;
   }

   drm_intel_bo_map(monitor->pipeline_stats_bo, false);
   uint64_t *start = monitor->pipeline_stats_bo->virtual;
   uint64_t *end = start + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint64_t));

   for (int i = 0; i < num_counters; i++) {
      monitor->pipeline_stats_results[i] = end[i] - start[i];
   }
   drm_intel_bo_unmap(monitor->pipeline_stats_bo);
   drm_intel_bo_unreference(monitor->pipeline_stats_bo);
   monitor->pipeline_stats_bo = NULL;
}

/******************************************************************************/

static bool
monitor_needs_oa(struct brw_context *brw,
                 struct gl_perf_monitor_object *m)
{
   return m->ActiveGroups[OA_COUNTERS];
}

/**
 * The amount of batch space it takes to emit an MI_REPORT_PERF_COUNT snapshot,
 * including the required PIPE_CONTROL flushes.
 *
 * Sandybridge is the worst case scenario: intel_batchbuffer_emit_mi_flush
 * expands to three PIPE_CONTROLs which are 4 DWords each.  We have to flush
 * before and after MI_REPORT_PERF_COUNT, so multiply by two.  Finally, add
 * the 3 DWords for MI_REPORT_PERF_COUNT itself.
 */
#define MI_REPORT_PERF_COUNT_BATCH_DWORDS (2 * (3 * 4) + 3)

/**
 * Emit an MI_REPORT_PERF_COUNT command packet.
 *
 * This writes the current OA counter values to buffer.
 */
static void
emit_mi_report_perf_count(struct brw_context *brw,
                          drm_intel_bo *bo,
                          uint32_t offset_in_bytes,
                          uint32_t report_id)
{
   assert(offset_in_bytes % 64 == 0);

   /* Make sure the commands to take a snapshot fits in a single batch. */
   intel_batchbuffer_require_space(brw, MI_REPORT_PERF_COUNT_BATCH_DWORDS * 4,
                                   RENDER_RING);
   int batch_used = brw->batch.used;

   /* Reports apparently don't always get written unless we flush first. */
   intel_batchbuffer_emit_mi_flush(brw);

   if (brw->gen == 5) {
      /* Ironlake requires two MI_REPORT_PERF_COUNT commands to write all
       * the counters.  The report ID is ignored in the second set.
       */
      BEGIN_BATCH(6);
      OUT_BATCH(GEN5_MI_REPORT_PERF_COUNT | GEN5_MI_COUNTER_SET_0);
      OUT_RELOC(bo,
                I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes);
      OUT_BATCH(report_id);

      OUT_BATCH(GEN5_MI_REPORT_PERF_COUNT | GEN5_MI_COUNTER_SET_1);
      OUT_RELOC(bo,
                I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes + 64);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else if (brw->gen == 6) {
      BEGIN_BATCH(3);
      OUT_BATCH(GEN6_MI_REPORT_PERF_COUNT);
      OUT_RELOC(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes | MI_COUNTER_ADDRESS_GTT);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else if (brw->gen == 7) {
      BEGIN_BATCH(3);
      OUT_BATCH(GEN6_MI_REPORT_PERF_COUNT);
      OUT_RELOC(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else {
      unreachable("Unsupported generation for performance counters.");
   }

   /* Reports apparently don't always get written unless we flush after. */
   intel_batchbuffer_emit_mi_flush(brw);

   (void) batch_used;
   assert(brw->batch.used - batch_used <= MI_REPORT_PERF_COUNT_BATCH_DWORDS * 4);
}

/**
 * Add a monitor to the global list of "unresolved monitors."
 *
 * Monitors are "unresolved" until the end MI_RPC OA counter snapshot has
 * been accumulated.
 *
 * Any monitors listed here may depend on counter snapshots within
 * bookend_bo, so if we run out of space we must iterate these monitors
 * to accumulate partial results before it's save to recycle bookend_bo.
 */
static void
add_to_unresolved_monitor_list(struct brw_context *brw,
                               struct brw_perf_monitor_object *monitor)
{
   if (brw->perfmon.unresolved_elements >=
       brw->perfmon.unresolved_array_size) {
      brw->perfmon.unresolved_array_size *= 2;
      brw->perfmon.unresolved = reralloc(brw, brw->perfmon.unresolved,
                                         struct brw_perf_monitor_object *,
                                         brw->perfmon.unresolved_array_size);
   }

   brw->perfmon.unresolved[brw->perfmon.unresolved_elements++] = monitor;
}

/**
 * Remove a monitor from the global list of "unresolved monitors." once
 * the end MI_RPC OA counter snapshot has been accumulated, or when
 * discarding unwanted monitor results.
 */
static void
drop_from_unresolved_monitor_list(struct brw_context *brw,
                                  struct brw_perf_monitor_object *monitor)
{
   for (int i = 0; i < brw->perfmon.unresolved_elements; i++) {
      if (brw->perfmon.unresolved[i] == monitor) {
         int last_elt = --brw->perfmon.unresolved_elements;

         if (i == last_elt) {
            brw->perfmon.unresolved[i] = NULL;
         } else {
            brw->perfmon.unresolved[i] = brw->perfmon.unresolved[last_elt];
         }

         /* If there are no unresolved monitors left that implies we can
          * recycle the bookend_bo, discarding any snapshots it held... */
         if (brw->perfmon.unresolved_elements == 0) {
            DBG("***Resetting bookend snapshots to 0\n");
            brw->perfmon.n_bookend_snapshots = 0;
         }
         return;
      }
   }
}

/* XXX: For BDW+ we'll need to check fuse registers */
static int
get_eu_count(uint32_t devid)
{
   if (IS_HSW_GT1(devid))
      return 10;
   if (IS_HSW_GT2(devid))
      return 20;
   if (IS_HSW_GT3(devid))
      return 40;
   assert(0);
}

static uint64_t
read_report_timestamp(struct brw_context *brw, uint32_t *report)
{
   struct brw_oa_counter *counter = &brw->perfmon.oa_counters[OA_GPU_TIMESTAMP];

   return counter->read(counter,
                        report,
                        NULL /* ignores report1 */,
                        NULL /* ignores accumulated */);
}

/**
 * Given pointers to starting and ending OA snapshots, add the deltas for each
 * counter to the results.
 */
static void
add_deltas(struct brw_context *brw,
           struct brw_perf_monitor_object *monitor,
           uint32_t *start, uint32_t *end)
{
#if 0
   fprintf(stderr, "Accumulating delta:\n");
   fprintf(stderr, "> Start timestamp = %" PRIu64 "\n", read_report_timestamp(brw, start));
   fprintf(stderr, "> End timestamp = %" PRIu64 "\n", read_report_timestamp(brw, end));
#endif

   for (int i = 0; i < MAX_OA_COUNTERS; i++) {
      struct brw_oa_counter *counter = &brw->perfmon.oa_counters[i];
      //uint64_t pre_accumulate;

      if (!counter->accumulate)
         continue;

      //pre_accumulate = monitor->oa_accumulator[counter->id];
      counter->accumulate(counter,
                          start, end,
                          monitor->oa_accumulator);
#if 0
      fprintf(stderr, "> Updated %s from %" PRIu64 " to %" PRIu64 "\n",
              counter->name, pre_accumulate,
              monitor->oa_accumulator[counter->id]);
#endif
   }

   monitor->accumulated = true;
}

/**
 * Accumulate OA counter results (partial or full) from a series of snapshots.
 *
 * N.B. We write snapshots for the beginning and end of a monitor into
 * monitor->oa_bo as well as snapshots at the start of each batch between
 * those points (into brw->perfmon.bookend_bo).
 *
 * These intermediate snapshots help to ensure we handle counter wrapping
 * correctly by being frequent enough to ensure we don't miss multiple
 * wrap arounds of a counter between snapshots.
 *
 * Note: In the future we will likely switch to collecting intermediate
 * snapshots via perf instead of explicitly issuing intermediate MI_RPC
 * commands.
 *
 * Note bookend_bo can be shared by multiple overlapping monitor objects and
 * its possible for us to run out of space in this BO whereby we have to
 * accumulate partial results for all currently unresolved monitors so that
 * the buffer can be recycled.
 *
 * Since the process of accumulating snapshot results always refers to two
 * snapshots that we can calculate a delta between, then in the case that we
 * accumulate partial results it's necessary to save the last snapshot
 * read from bookend_bo to brw->perfmon.gather_continue_snapshotp[] to serve
 * as a reference point when we continue to accumulate results later.
 * (Actually capturing this snapshot is handled in wrap_bookend_bo(), but
 * we're careful to refer to it as necessary here.)
 */
static void
accumulate_oa_snapshots(struct brw_context *brw,
                        struct brw_perf_monitor_object *monitor,
                        uint32_t *bookend_buffer)
{
   struct gl_perf_monitor_object *m = &monitor->base;

   assert(!monitor->resolved);
   assert(monitor->oa_bo != NULL);
   assert(monitor->oa_bo->virtual != NULL);

   uint32_t *monitor_buffer = monitor->oa_bo->virtual;

   const int snapshot_size = brw->perfmon.entries_per_oa_snapshot;

   /* If we have any bookend snapshots then count the contribution of those
    * first.
    */

   int n_be_snapshots = (monitor->bookend_snapshots_end -
                         monitor->bookend_snapshots_begin);

   if (n_be_snapshots) {
      uint32_t *start;
      uint32_t *end;

      /* If we've accumulated any partial results, because we ran out of
       * space in bookend_bo then we need to refer to
       * perfmon.gather_continue_snapshot[] for our next delta...
       */
      if (monitor->accumulated) {
         assert(monitor->bookend_snapshots_begin == 0);
         start = brw->perfmon.gather_continue_snapshot;
         end = bookend_buffer;
      } else {
         start = monitor_buffer;
         end = bookend_buffer + snapshot_size * monitor->bookend_snapshots_begin;

         if (start[0] != REPORT_ID) {
            //fprintf(stderr, "Monitor's beginning OA report was lost!");
            start = end;
         }
      }

      for (int i = 0; i < n_be_snapshots; i++) {

         /* It's possible that the MI_RPC snapshot was lost if it collided
          * with a previous, in-flight MI_RPC command. In this case we
          * assume it's fine to ignore the missing snapshot since it implies
          * that there would be no significant delta between the snapshots
          * so we aren't going to miss a counter wrapping.
          *
          * XXX: consider using a dynamic report id to avoid false-positives
          * here when recycling bookend_bo.
          */
         if (start[0] == REPORT_ID && end[0] == REPORT_ID)
            add_deltas(brw, monitor, start, end);

         start = end;
         end += snapshot_size;
      }
   }

   /* Resolve the monitor and accumulate the end snapshot if possible... */
   if (m->Ended) {
      uint32_t *start;
      uint32_t *end = monitor_buffer + (SECOND_SNAPSHOT_OFFSET_IN_BYTES /
                                        sizeof(uint32_t));

      if (n_be_snapshots) {
         /* Use last snapshot accumulated above */
         start = bookend_buffer + snapshot_size * (monitor->bookend_snapshots_end - 1);
      } else if (monitor->accumulated) {
         start = brw->perfmon.gather_continue_snapshot;
      } else {
         start = monitor_buffer;
      }

      if (end[0] != REPORT_ID) {
         fprintf(stderr, "Monitor's end OA report was lost!");
         end = start;
      }

      if (start[0] == REPORT_ID && end[0] == REPORT_ID)
         add_deltas(brw, monitor, start, end);

      monitor->resolved = true;

      /* The monitor's OA result is now resolved. */
      DBG("Marking %d resolved - results gathered\n", m->Name);
      drop_from_unresolved_monitor_list(brw, monitor);
   }
}

/**
 * Handle running out of space in the bookend BO.
 *
 * When we run out of space in the bookend BO, we need to gather up partial
 * results for every unresolved monitor.  This allows us to free the snapshot
 * data in bookend_bo, freeing up the space for reuse.  We call this "wrapping."
 *
 * This will completely compute the result for any unresolved monitors that
 * have ended.
 */
static void
wrap_bookend_bo(struct brw_context *brw)
{
   DBG("****Wrap bookend BO****\n");

   /* Note that wrapping will only occur at the start of a batch, since that's
    * where we reserve space.  So the current batch won't reference bookend_bo
    * or any monitor BOs.  This means we don't need to worry about
    * synchronization.
    */
   assert(brw->perfmon.oa_users > 0);

   drm_intel_bo_map(brw->perfmon.bookend_bo, false);
   uint32_t *bookend_buffer = brw->perfmon.bookend_bo->virtual;
   for (int i = 0; i < brw->perfmon.unresolved_elements; i++) {
      struct brw_perf_monitor_object *monitor = brw->perfmon.unresolved[i];
      struct gl_perf_monitor_object *m = &monitor->base;

      /* If the monitor has ended and we're just waiting for the last MI_RPC
       * snapshot to land then we don't want to accumulate any bookend
       * snapshots made after monitoring ended... */
      if (!m->Ended)
         monitor->bookend_snapshots_end = brw->perfmon.n_bookend_snapshots;

      drm_intel_bo_map(monitor->oa_bo, false);
      accumulate_oa_snapshots(brw, monitor, bookend_buffer);
      drm_intel_bo_unmap(monitor->oa_bo);

      monitor->bookend_snapshots_begin = 0;
      monitor->bookend_snapshots_end = 0;

      if (monitor->resolved) {
         --i; /* account for modifying the list while iterating */
      }
   }

   const int snapshot_size = brw->perfmon.entries_per_oa_snapshot;

   /* Since the process of accumulating snapshot results always refers to
    * two snapshots that we can calculate a delta between, that means it's
    * necessary to copy the last bookend_bo snapshot to
    * brw->perfmon.gather_continue_snapshotp[] here to serve as a reference
    * point when we continue to accumulate results later...
    */
   memcpy(brw->perfmon.gather_continue_snapshot,
          bookend_buffer + snapshot_size * (brw->perfmon.n_bookend_snapshots - 1),
          snapshot_size * sizeof(uint32_t));

   drm_intel_bo_unmap(brw->perfmon.bookend_bo);

   brw->perfmon.n_bookend_snapshots = 0;
}

/* This is fairly arbitrary; the trade off is memory usage vs. extra overhead
 * from wrapping.  On Gen7, 32768 should be enough for for 128 snapshots before
 * wrapping (since each is 256 bytes).
 */
#define BOOKEND_BO_SIZE_BYTES 32768

static bool
has_space_for_bookend_snapshot(struct brw_context *brw)
{
   int snapshot_bytes = brw->perfmon.entries_per_oa_snapshot * sizeof(uint32_t);
   int total_snapshots = brw->perfmon.n_bookend_snapshots + 1;

   return total_snapshots * snapshot_bytes < BOOKEND_BO_SIZE_BYTES;
}

/**
 * Write an OA counter snapshot to bookend_bo.
 */
static void
emit_bookend_snapshot(struct brw_context *brw)
{
   int snapshot_bytes = brw->perfmon.entries_per_oa_snapshot * sizeof(uint32_t);
   int offset_in_bytes = brw->perfmon.n_bookend_snapshots * snapshot_bytes;

   emit_mi_report_perf_count(brw, brw->perfmon.bookend_bo, offset_in_bytes,
                             REPORT_ID);
   ++brw->perfmon.n_bookend_snapshots;
}

/******************************************************************************/

static uint64_t
read_file_uint64 (const char *file)
{
        char buf[32];
        int fd, n;

        fd = open(file, 0);
        if (fd < 0)
                return 0;
        n = read(fd, buf, sizeof (buf) - 1);
        close(fd);
        if (n < 0)
                return 0;

        buf[n] = '\0';
        return strtoull(buf, 0, 0);
}

static uint64_t
lookup_i915_oa_id (void)
{
   return read_file_uint64 ("/sys/bus/event_source/devices/i915_oa/type");
}

static long
perf_event_open (struct perf_event_attr *hw_event,
                 pid_t pid,
                 int cpu,
                 int group_fd,
                 unsigned long flags)
{
   return syscall (__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int
open_i915_oa_event (uint64_t report_format,
                    int period_exponent,
                    int drm_fd,
                    uint32_t ctx_id)
{
   struct perf_event_attr attr;
   int event_fd;

   memset(&attr, 0, sizeof (struct perf_event_attr));
   attr.size = sizeof (struct perf_event_attr);
   attr.type = lookup_i915_oa_id();

   attr.config |= report_format;
   attr.config |= (uint64_t)period_exponent << I915_PERF_OA_TIMER_EXPONENT_SHIFT;

   attr.config |= I915_PERF_OA_SINGLE_CONTEXT_ENABLE;
   attr.config |= ctx_id & I915_PERF_OA_CTX_ID_MASK;
   attr.config1 = drm_fd;

   attr.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_READ | PERF_SAMPLE_RAW;
   attr.disabled = 1;
   attr.sample_period = 0;

   //To avoid needing CAP_SYS_ADMIN...
   //attr.exclude_kernel = 1;

   event_fd = perf_event_open(&attr,
                              -1,  /* pid */
                              0, /* cpu */
                              -1, /* group fd */
                              PERF_FLAG_FD_CLOEXEC); /* flags */
   if (event_fd == -1) {
      DBG("WARNING: Error opening i915_oa perf event: %m\n");
      return -1;
   }

   return event_fd;
}

/**
 * Initialize a monitor to sane starting state; throw away old buffers.
 */
static void
reinitialize_perf_monitor(struct brw_context *brw,
                          struct brw_perf_monitor_object *monitor)
{
   if (monitor->oa_bo) {
      drm_intel_bo_unreference(monitor->oa_bo);
      monitor->oa_bo = NULL;
   }

   /* Since the results are now invalid, we don't need to hold on to any
    * snapshots in bookend_bo.  The monitor is effectively "resolved."
    */
   drop_from_unresolved_monitor_list(brw, monitor);
   monitor->resolved = false;

   monitor->bookend_snapshots_begin = brw->perfmon.n_bookend_snapshots;
   monitor->bookend_snapshots_end = brw->perfmon.n_bookend_snapshots;

   memset(monitor->oa_accumulator, 0, sizeof(monitor->oa_accumulator));
   monitor->accumulated = false;

   if (monitor->pipeline_stats_bo) {
      drm_intel_bo_unreference(monitor->pipeline_stats_bo);
      monitor->pipeline_stats_bo = NULL;
   }

   free(monitor->pipeline_stats_results);
   monitor->pipeline_stats_results = NULL;
}

/**
 * Driver hook for glBeginPerformanceMonitorAMD().
 */
static GLboolean
brw_begin_perf_monitor(struct gl_context *ctx,
                       struct gl_perf_monitor_object *m)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_monitor_object *monitor = brw_perf_monitor(m);

   DBG("Begin(%d)\n", m->Name);

   reinitialize_perf_monitor(brw, monitor);

   if (monitor_needs_oa(brw, m)) {
      /* If the OA counters aren't already on, enable them. */
      if (brw->perfmon.perf_oa_event_fd == -1) {
         __DRIscreen *screen = brw->intelScreen->driScrnPriv;
         int period_exponent;
         int fd;

         /* The timestamp for HSW+ increments every 80ns
          *
          * The period_exponent gives a sampling period as follows:
          *   sample_period = 80ns * 2^(period_exponent + 1)
          *
          * FIXME: we need to choose a short enough period to catch
          * counters wrapping.
          *
          * The overflow period for Haswell can be calculated as:
          *
          * 2^32 / (n_eus * max_gen_freq * 2)
          * (E.g. 40 EUs @ 1GHz = ~53ms)
          *
          * Currently we just sample ~ every 5 milliseconds...
          */
         //period_exponent = 15;
         period_exponent = 18;

         fd = open_i915_oa_event(I915_PERF_OA_FORMAT_A45_B8_C8_HSW,
                                 period_exponent,
                                 screen->fd, /* drm fd */
                                 brw->hw_ctx->ctx_id);
         if (fd == -1)
            return GL_FALSE; /* XXX: do we need to set GL error state? */

         brw->perfmon.perf_oa_event_fd = fd;
      }

      if (brw->perfmon.oa_users == 0 &&
          ioctl(brw->perfmon.perf_oa_event_fd, PERF_EVENT_IOC_ENABLE, 0) < 0)
      {
         DBG("WARNING: Error enabling i915_oa perf event: %m\n");
         return GL_FALSE; /* XXX: do we need to set GL error state? */
      }

      /* If the global OA bookend BO doesn't exist, allocate it.  This should
       * only happen once, but we delay until BeginPerfMonitor time to avoid
       * wasting memory for contexts that don't use performance monitors.
       */
      if (!brw->perfmon.bookend_bo) {
         brw->perfmon.bookend_bo = drm_intel_bo_alloc(brw->bufmgr,
                                                      "OA bookend BO",
                                                      BOOKEND_BO_SIZE_BYTES, 64);
      }

      monitor->oa_bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. monitor OA bo", 4096, 64);
#ifdef DEBUG
      /* Pre-filling the BO helps debug whether writes landed. */
      drm_intel_bo_map(monitor->oa_bo, true);
      memset((char *) monitor->oa_bo->virtual, 0xff, 4096);
      drm_intel_bo_unmap(monitor->oa_bo);
#endif

      /* Take a starting OA counter snapshot. */
      emit_mi_report_perf_count(brw, monitor->oa_bo, 0, REPORT_ID);

      /* Add the monitor to the unresolved list. */
      add_to_unresolved_monitor_list(brw, monitor);

      ++brw->perfmon.oa_users;
      ++brw->perfmon.open_oa_monitors;
   }

   if (monitor_needs_statistics_registers(brw, m)) {
      monitor->pipeline_stats_bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. monitor stats bo", 4096, 64);

      /* Take starting snapshots. */
      snapshot_statistics_registers(brw, monitor, 0);
   }

   return true;
}

/**
 * Driver hook for glEndPerformanceMonitorAMD().
 */
static void
brw_end_perf_monitor(struct gl_context *ctx,
                     struct gl_perf_monitor_object *m)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_monitor_object *monitor = brw_perf_monitor(m);

   DBG("End(%d)\n", m->Name);

   if (monitor_needs_oa(brw, m)) {
      /* Take an ending OA counter snapshot. */
      emit_mi_report_perf_count(brw, monitor->oa_bo,
                                SECOND_SNAPSHOT_OFFSET_IN_BYTES, REPORT_ID);

      --brw->perfmon.open_oa_monitors;

      /* Now that the monitor has ended, (but can't be resolved until we
       * have read the last MI_RPC snapshot) we mark the last bookend
       * snapshot associated with this monitor so we won't gather further
       * snapshots that might be made before this monitor is resolved.
       */
      monitor->bookend_snapshots_end = brw->perfmon.n_bookend_snapshots;
   }

   if (monitor_needs_statistics_registers(brw, m)) {
      /* Take ending snapshots. */
      snapshot_statistics_registers(brw, monitor,
                                    SECOND_SNAPSHOT_OFFSET_IN_BYTES);
   }
}

/**
 * Reset a performance monitor, throwing away any results.
 */
static void
brw_reset_perf_monitor(struct gl_context *ctx,
                       struct gl_perf_monitor_object *m)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_monitor_object *monitor = brw_perf_monitor(m);

   reinitialize_perf_monitor(brw, monitor);

   if (m->Active) {
      brw_begin_perf_monitor(ctx, m);
   }
}

/**
 * Is a performance monitor result available?
 */
static GLboolean
brw_is_perf_monitor_result_available(struct gl_context *ctx,
                                     struct gl_perf_monitor_object *m)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_monitor_object *monitor = brw_perf_monitor(m);

   bool oa_available = true;
   bool stats_available = true;

   if (monitor_needs_oa(brw, m)) {
      oa_available = !monitor->oa_bo ||
         (!drm_intel_bo_references(brw->batch.bo, monitor->oa_bo) &&
          !drm_intel_bo_busy(monitor->oa_bo));
   }

   if (monitor_needs_statistics_registers(brw, m)) {
      stats_available = !monitor->pipeline_stats_bo ||
         (!drm_intel_bo_references(brw->batch.bo, monitor->pipeline_stats_bo) &&
          !drm_intel_bo_busy(monitor->pipeline_stats_bo));
   }

   return oa_available && stats_available;
}

/**
 * Get the performance monitor result.
 */
static void
brw_get_perf_monitor_result(struct gl_context *ctx,
                            struct gl_perf_monitor_object *m,
                            GLsizei data_size,
                            GLuint *data,
                            GLint *bytes_written)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_monitor_object *monitor = brw_perf_monitor(m);
   uint8_t *p = (uint8_t *)data;

   assert(brw_is_perf_monitor_result_available(ctx, m));

   DBG("GetResult(%d)\n", m->Name);
   brw_dump_perf_monitors(brw);

   /* This hook should only be called when results are available. */
   assert(m->Ended);

   /* Copy data to the supplied array (data).
    *
    * The output data format is: <group ID, counter ID, value> for each
    * active counter.  The API allows counters to appear in any order.
    */
   GLsizei offset = 0;

   if (monitor_needs_oa(brw, m)) {
      int n_oa_counters = ctx->PerfMonitor.Groups[OA_COUNTERS].NumCounters;

      drm_intel_bo_map(monitor->oa_bo, false);

      if (!monitor->resolved) {
         /* Since the result is available, all the necessary snapshots will
          * have been written to the bookend BO.  If other monitors are
          * active, the bookend BO may be busy or referenced by the current
          * batch, but only for writing snapshots related to those other
          * monitors which we don't care about.
          *
          * Using an unsynchronized mapping avoids stalling for an
          * indeterminate amount of time.
          */
         drm_intel_gem_bo_map_unsynchronized(brw->perfmon.bookend_bo);

         accumulate_oa_snapshots(brw, monitor, brw->perfmon.bookend_bo->virtual);

         drm_intel_bo_unmap(brw->perfmon.bookend_bo);

         assert(monitor->resolved);
      }

      /* Disabling the i915_oa event will effectively disable the OA
       * counters.  Note it's important to be sure there are no outstanding
       * MI_RPC commands at this point since they could stall the CS
       * indefinitely once OACONTROL is disabled.
       */
      --brw->perfmon.oa_users;
      if (brw->perfmon.oa_users == 0 &&
          ioctl(brw->perfmon.perf_oa_event_fd, PERF_EVENT_IOC_DISABLE, 0) < 0)
      {
         DBG("WARNING: Error disabling i915_oa perf event: %m\n");
      }

      uint32_t *start = monitor->oa_bo->virtual;
      uint32_t *end = start + (SECOND_SNAPSHOT_OFFSET_IN_BYTES /
                               sizeof(uint32_t));

      for (int i = 0; i < n_oa_counters; i++) {
         enum brw_oa_counter_id id = brw->perfmon.oa_counter_map[i];
         struct brw_oa_counter *counter = &brw->perfmon.oa_counters[id];
         const struct gl_perf_monitor_counter *gl_counter =
            &ctx->PerfMonitor.Groups[OA_COUNTERS].Counters[i];

         /* We always capture all the OA counters, but the application may
          * have only asked for a subset.  Skip unwanted counters.
          */
         if (!BITSET_TEST(m->ActiveCounters[OA_COUNTERS], i))
            continue;

         *((uint32_t *)p) = OA_COUNTERS;
         p += 4;
         *((uint32_t *)p) = i;
         p += 4;

         switch(gl_counter->Type)
         {
         case GL_UNSIGNED_INT:
            *((uint32_t *)p) =
               counter->read(counter, start, end, monitor->oa_accumulator);
            p += 4;
            break;
         case GL_PERCENTAGE_AMD:
            /* TODO: Enable floating point precision instead of casting an int */
            *((float *)p) =
               counter->read(counter, start, end, monitor->oa_accumulator);
#if 0
            fprintf(stderr, "DEBUG PERCENTAGE: %20s = %3d: raw ref counter = %-3d delta=%-12"PRIu64 " start=%-11"PRIu32 " end=%-11"PRIu32"\n",
                    gl_counter->Name, (int)*((float *)p),
                    counter->report_offset,
                    monitor->oa_accumulator[counter->id],
                    start[counter->report_offset], end[counter->report_offset]);
#endif
            p += 4;
            break;
         case GL_UNSIGNED_INT64_AMD:
            *((uint64_t *)p) =
               counter->read(counter, start, end, monitor->oa_accumulator);
            p += 8;
            break;
         }
      }

      drm_intel_bo_unmap(monitor->oa_bo);
   }

   if (monitor_needs_statistics_registers(brw, m)) {
      const int num_counters =
         ctx->PerfMonitor.Groups[PIPELINE_STATS_COUNTERS].NumCounters;

      if (!monitor->pipeline_stats_results) {
         gather_statistics_results(brw, monitor);

         /* Check if we did really get the results */
         if (!monitor->pipeline_stats_results) {
            if (bytes_written) {
               *bytes_written = 0;
            }
            return;
         }
      }

      for (int i = 0; i < num_counters; i++) {
         if (BITSET_TEST(m->ActiveCounters[PIPELINE_STATS_COUNTERS], i)) {

            *((uint32_t *)p) = PIPELINE_STATS_COUNTERS;
            p += 4;
            *((uint32_t *)p) = i;
            p += 4;
            *((uint64_t *)p) = monitor->pipeline_stats_results[i];
            p += 8;
         }
      }
   }

   if (bytes_written)
      *bytes_written = p - (uint8_t *)data;
}

/**
 * Create a new performance monitor object.
 */
static struct gl_perf_monitor_object *
brw_new_perf_monitor(struct gl_context *ctx)
{
   return calloc(1, sizeof(struct brw_perf_monitor_object));
}

/**
 * Delete a performance monitor object.
 */
static void
brw_delete_perf_monitor(struct gl_context *ctx, struct gl_perf_monitor_object *m)
{
   struct brw_perf_monitor_object *monitor = brw_perf_monitor(m);
   DBG("Delete(%d)\n", m->Name);
   reinitialize_perf_monitor(brw_context(ctx), monitor);
   free(monitor);
}

/******************************************************************************/

/**
 * Called at the start of every render ring batch.
 *
 * Enable OA counters and emit the "start of batchbuffer" bookend OA snapshot.
 * Since it's a new batch, there will be plenty of space for the commands.
 */
void
brw_perf_monitor_new_batch(struct brw_context *brw)
{
   assert(brw->batch.ring == RENDER_RING);
   assert(brw->gen < 6 || brw->batch.used == 0);

   if (brw->perfmon.open_oa_monitors == 0)
      return;

   /* Make sure bookend_bo has enough space for a pair of snapshots.  If
    * not, "wrap" the BO: gather up any results so far, and start from the
    * beginning of the buffer.
    */
   if (!has_space_for_bookend_snapshot(brw))
      wrap_bookend_bo(brw);

   DBG("Bookend Begin Snapshot (%d)\n", brw->perfmon.n_bookend_snapshots);
   emit_bookend_snapshot(brw);
}

/**
 * Called at the end of every render ring batch.
 */
void
brw_perf_monitor_finish_batch(struct brw_context *brw)
{
   assert(brw->batch.ring == RENDER_RING);

   /* TODO: remove this:
    *
    * We no longer capture a snapshot at the end of a BB since we assume
    * that the end of one BB will be close enough to the start of the next
    * that we won't accumulate a very significant delta and reducing the
    * number of snapshots we request may help avoid MI_RPC collisions that
    * can result in lost reports.
    *
    * Eventually, instead of requesting snapshots at BB boundaries we will
    * collect the periodic snapshots from perf.
    */
}

/******************************************************************************/

static uint64_t
read_oa_counter_raw_cb(struct brw_oa_counter *counter,
                       uint32_t *report0,
                       uint32_t *report1,
                       uint64_t *accumulator)
{
   return report0[counter->report_offset];
}

static void
accumulate_uint32_cb(struct brw_oa_counter *counter,
                     uint32_t *report0,
                     uint32_t *report1,
                     uint64_t *accumulator)
{
   enum brw_oa_counter_id id = counter->id;

   /* XXX: BRW introduces 40bit counters where we'll need to be a bit
    * more careful considering wrapping */
   accumulator[id] += (uint32_t)(report1[counter->report_offset] -
                                 report0[counter->report_offset]);
}

static uint64_t
read_oa_timestamp_cb(struct brw_oa_counter *counter,
                     uint32_t *report0,
                     uint32_t *report1,
                     uint64_t *accumulator)
{
   uint32_t time0 = report0[counter->report_offset];
   uint32_t time1 = report0[counter->report_offset + 1];
   uint64_t timestamp = (uint64_t)time1 << 32 | time0;

   /* The least significant timestamp bit represents 80ns on Haswell */
   timestamp *= 80;
   timestamp /= 1000; /* usecs */

   return timestamp;
}

static struct brw_oa_counter *
add_raw_oa_counter(struct brw_context *brw,
                   enum brw_oa_counter_id id, int report_offset)
{
   struct brw_oa_counter *counter;

   assert(id < MAX_OA_COUNTERS);

   counter = &brw->perfmon.oa_counters[id];

   counter->name = "raw";
   counter->id = id;
   counter->report_offset = report_offset;
   counter->accumulate = accumulate_uint32_cb;
   counter->read = read_oa_counter_raw_cb;

   return counter;
}

static uint64_t
read_oa_counter_normalized_by_gpu_duration_cb(struct brw_oa_counter *counter,
                                              uint32_t *report0,
                                              uint32_t *report1,
                                              uint64_t *accumulated)
{
   uint64_t delta = accumulated[counter->id];
   uint64_t clk_delta = accumulated[OA_GPU_CORE_CLOCK];

   if (!clk_delta)
      return 0;

   return delta * 100 / clk_delta;
}

static struct brw_oa_counter *
add_oa_counter_normalised_by_gpu_duration(struct brw_context *brw,
                                          enum brw_oa_counter_id id,
                                          const char *name,
                                          int report_offset)
{
   struct brw_oa_counter *counter = add_raw_oa_counter(brw, id, report_offset);

   counter->name = name;
   counter->read = read_oa_counter_normalized_by_gpu_duration_cb;

   return counter;
}

static uint64_t
read_oa_counter_normalized_by_eu_duration_cb(struct brw_oa_counter *counter,
                                             uint32_t *report0,
                                             uint32_t *report1,
                                             uint64_t *accumulated)
{
   uint64_t delta = accumulated[counter->id];
   uint64_t clk_delta = accumulated[OA_GPU_CORE_CLOCK];

   delta /= eu_count;

   if (!clk_delta)
      return 0;

   return delta * 100 / clk_delta;
}

static struct brw_oa_counter *
add_oa_counter_normalised_by_eu_duration(struct brw_context *brw,
                                         enum brw_oa_counter_id id,
                                         const char *name,
                                         int report_offset)
{
   struct brw_oa_counter *counter = add_raw_oa_counter(brw, id, report_offset);

   counter->name = name;
   counter->read = read_oa_counter_normalized_by_eu_duration_cb;

   return counter;
}

static uint64_t
read_av_thread_cycles_counter_cb (struct brw_oa_counter *counter,
                                  uint32_t *report0,
                                  uint32_t *report1,
                                  uint64_t *accumulated)
{
   uint64_t delta = accumulated[counter->id];
   uint64_t spawned = accumulated[counter->config];

   if (!spawned)
      return 0;

   return delta / spawned;
}

static struct brw_oa_counter *
add_average_thread_cycles_oa_counter (struct brw_context *brw,
                                      enum brw_oa_counter_id id,
                                      const char *name,
                                      int count_report_offset,
                                      int denominator_report_offset)
{
   struct brw_oa_counter *counter = add_raw_oa_counter(brw, id, count_report_offset);

   counter->name = name;
   counter->read = read_av_thread_cycles_counter_cb;
   counter->config = denominator_report_offset;

   return counter;
}

static void
init_hsw_oa_counters(struct brw_context *brw)
{
   struct brw_oa_counter *c;
   int a_offset = 3; /* A0 */
   int b_offset = a_offset + 45; /* B0 */

   c = add_raw_oa_counter(brw, OA_GPU_TIMESTAMP, 1);
   c->read = read_oa_timestamp_cb;

   add_raw_oa_counter(brw, OA_AGGREGATE_CORE_ARRAYS_ACTIVE, a_offset);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_EU_ACTIVE_PERCENTAGE,
                                            "EU Active %",
                                            a_offset);
   add_raw_oa_counter(brw, OA_AGGREGATE_CORE_ARRAYS_STALLED, a_offset + 1);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_EU_STALLED_PERCENTAGE,
                                            "EU Stalled %",
                                            a_offset + 1);
   add_raw_oa_counter(brw, OA_VS_ACTIVE_TIME, a_offset + 2);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_VS_EU_ACTIVE_PERCENTAGE,
                                            "VS EU Active %",
                                            a_offset + 2);
   add_raw_oa_counter(brw, OA_VS_STALL_TIME, a_offset + 3);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_VS_EU_STALLED_PERCENTAGE,
                                            "VS EU Stalled %",
                                            a_offset + 3);
   add_raw_oa_counter(brw, OA_NUM_VS_THREADS_LOADED, a_offset + 5);

   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_VS_THREAD_CYCLES,
                                        "Av. cycles per VS thread",
                                        a_offset + 2,
                                        a_offset + 5);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_STALLED_VS_THREAD_CYCLES,
                                        "Av. stalled cycles per VS thread",
                                        a_offset + 3,
                                        a_offset + 5);

   add_raw_oa_counter(brw, OA_HS_ACTIVE_TIME, a_offset + 7);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_HS_EU_ACTIVE_PERCENTAGE,
                                            "HS EU Active %",
                                            a_offset + 7);
   add_raw_oa_counter(brw, OA_HS_STALL_TIME, a_offset + 8);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_HS_EU_STALLED_PERCENTAGE,
                                            "HS EU Stalled %",
                                            a_offset + 8);
   add_raw_oa_counter(brw, OA_NUM_HS_THREADS_LOADED, a_offset + 10);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_HS_THREAD_CYCLES,
                                        "Av. cycles per HS thread",
                                        a_offset + 7,
                                        a_offset + 10);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_STALLED_HS_THREAD_CYCLES,
                                        "Av. stalled cycles per HS thread",
                                        a_offset + 8,
                                        a_offset + 10);

   add_raw_oa_counter(brw, OA_DS_ACTIVE_TIME, a_offset + 12);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_DS_EU_ACTIVE_PERCENTAGE,
                                            "DS EU Active %",
                                            a_offset + 12);
   add_raw_oa_counter(brw, OA_DS_STALL_TIME, a_offset + 13);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_DS_EU_STALLED_PERCENTAGE,
                                            "DS EU Stalled %",
                                            a_offset + 13);
   add_raw_oa_counter(brw, OA_NUM_DS_THREADS_LOADED, a_offset + 15);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_DS_THREAD_CYCLES,
                                        "Av. cycles per DS thread",
                                        a_offset + 12,
                                        a_offset + 15);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_STALLED_DS_THREAD_CYCLES,
                                        "Av. stalled cycles per DS thread",
                                        a_offset + 13,
                                        a_offset + 15);

   add_raw_oa_counter(brw, OA_CS_ACTIVE_TIME, a_offset + 17);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_CS_EU_ACTIVE_PERCENTAGE,
                                            "CS EU Active %",
                                            a_offset + 17);
   add_raw_oa_counter(brw, OA_CS_STALL_TIME, a_offset + 18);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_CS_EU_STALLED_PERCENTAGE,
                                            "CS EU Stalled %",
                                            a_offset + 18);
   add_raw_oa_counter(brw, OA_NUM_CS_THREADS_LOADED, a_offset + 20);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_CS_THREAD_CYCLES,
                                        "Av. cycles per CS thread",
                                        a_offset + 17,
                                        a_offset + 20);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_STALLED_CS_THREAD_CYCLES,
                                        "Av. stalled cycles per CS thread",
                                        a_offset + 18,
                                        a_offset + 20);


   add_raw_oa_counter(brw, OA_GS_ACTIVE_TIME, a_offset + 22);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_GS_EU_ACTIVE_PERCENTAGE,
                                            "GS EU Active %",
                                            a_offset + 22);
   add_raw_oa_counter(brw, OA_GS_STALL_TIME, a_offset + 23);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_GS_EU_STALLED_PERCENTAGE,
                                            "GS EU Stalled %",
                                            a_offset + 23);
   add_raw_oa_counter(brw, OA_NUM_GS_THREADS_LOADED, a_offset + 25);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_GS_THREAD_CYCLES,
                                        "Av. cycles per GS thread",
                                        a_offset + 22,
                                        a_offset + 25);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_STALLED_GS_THREAD_CYCLES,
                                        "Av. stalled cycles per GS thread",
                                        a_offset + 23,
                                        a_offset + 25);


   add_raw_oa_counter(brw, OA_PS_ACTIVE_TIME, a_offset + 27);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_PS_EU_ACTIVE_PERCENTAGE,
                                            "PS EU Active %",
                                            a_offset + 27);
   add_raw_oa_counter(brw, OA_PS_STALL_TIME, a_offset + 28);
   add_oa_counter_normalised_by_eu_duration(brw,
                                            OA_PS_EU_STALLED_PERCENTAGE,
                                            "PS EU Stalled %",
                                            a_offset + 28);
   add_raw_oa_counter(brw, OA_NUM_PS_THREADS_LOADED, a_offset + 30);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_PS_THREAD_CYCLES,
                                        "Av. cycles per PS thread",
                                        a_offset + 27,
                                        a_offset + 30);
   add_average_thread_cycles_oa_counter(brw,
                                        OA_AVERAGE_STALLED_PS_THREAD_CYCLES,
                                        "Av. stalled cycles per PS thread",
                                        a_offset + 28,
                                        a_offset + 30);

   add_raw_oa_counter(brw, OA_HIZ_FAST_Z_PASSING, a_offset + 32);
   add_raw_oa_counter(brw, OA_HIZ_FAST_Z_FAILING, a_offset + 33);

   add_raw_oa_counter(brw, OA_SLOW_Z_FAILING, a_offset + 35);

   /* XXX: caveat: it's 2x real No. when PS has 2 output colors */
   add_raw_oa_counter(brw, OA_PIXEL_KILL_COUNT, a_offset + 36);

   add_raw_oa_counter(brw, OA_ALPHA_TEST_FAILED, a_offset + 37);
   add_raw_oa_counter(brw, OA_POST_PS_STENCIL_TEST_FAILED, a_offset + 38);
   add_raw_oa_counter(brw, OA_POST_PS_Z_TEST_FAILED, a_offset + 39);

   add_raw_oa_counter(brw, OA_RENDER_TARGET_WRITES, a_offset + 40);

   /* XXX: there are several conditions where this doesn't increment... */
   add_raw_oa_counter(brw, OA_RENDER_ENGINE_BUSY, a_offset + 41);
   add_oa_counter_normalised_by_gpu_duration(brw,
                                             OA_RENDER_BUSY_PERCENTAGE,
                                             "Render Engine Busy %",
                                             a_offset + 41);

   add_raw_oa_counter(brw, OA_VS_BOTTLENECK, a_offset + 42);
   add_raw_oa_counter(brw, OA_GS_BOTTLENECK, a_offset + 43);
   add_raw_oa_counter(brw, OA_GPU_CORE_CLOCK, b_offset);
}

void
brw_init_performance_monitors(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;

   ctx->Driver.NewPerfMonitor = brw_new_perf_monitor;
   ctx->Driver.DeletePerfMonitor = brw_delete_perf_monitor;
   ctx->Driver.BeginPerfMonitor = brw_begin_perf_monitor;
   ctx->Driver.EndPerfMonitor = brw_end_perf_monitor;
   ctx->Driver.ResetPerfMonitor = brw_reset_perf_monitor;
   ctx->Driver.IsPerfMonitorResultAvailable = brw_is_perf_monitor_result_available;
   ctx->Driver.GetPerfMonitorResult = brw_get_perf_monitor_result;

   if (brw->gen == 6) {
      ctx->PerfMonitor.Groups = gen6_groups;
      ctx->PerfMonitor.NumGroups = ARRAY_SIZE(gen6_groups);
      brw->perfmon.entries_per_oa_snapshot = 0;
      brw->perfmon.n_exposed_oa_counters = 0;
      brw->perfmon.oa_counter_map = NULL;
      brw->perfmon.statistics_registers = gen6_statistics_register_addresses;
   } else if (brw->gen == 7) {
      ctx->PerfMonitor.Groups = gen7_groups;
      ctx->PerfMonitor.NumGroups = ARRAY_SIZE(gen7_groups);
      brw->perfmon.entries_per_oa_snapshot = 64;
      brw->perfmon.n_exposed_oa_counters = ARRAY_SIZE(gen7_normalized_oa_counters); /* TODO: remove */
      brw->perfmon.oa_counter_map = gen7_oa_counter_map;
      brw->perfmon.statistics_registers = gen7_statistics_register_addresses;
   }

   brw->perfmon.unresolved =
      ralloc_array(brw, struct brw_perf_monitor_object *, 1);
   brw->perfmon.unresolved_elements = 0;
   brw->perfmon.unresolved_array_size = 1;

   brw->perfmon.perf_oa_event_fd = -1;

   memset(brw->perfmon.oa_counters, 0, sizeof(brw->perfmon.oa_counters));
   init_hsw_oa_counters(brw);

   eu_count = get_eu_count(brw->intelScreen->deviceID);
}

void
brw_destroy_performance_monitors(struct brw_context *brw)
{
   if (brw->perfmon.perf_oa_event_fd != -1) {
      close(brw->perfmon.perf_oa_event_fd);
      brw->perfmon.perf_oa_event_fd = -1;
   }

   /* FIXME: clean up bookend bo etc */
}
