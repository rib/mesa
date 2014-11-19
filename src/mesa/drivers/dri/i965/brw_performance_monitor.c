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
   int current_report_id;

   /**
    * We collect periodic counter snapshots via perf so we can account
    * for counter overflow and this is a pointer into the circular
    * perf buffer for collecting snapshots that lie within the begin-end
    * bounds of this monitor.
    */
   unsigned int oa_tail;

   /**
    * Storage for OA results accumulated so far.
    *
    * An array indexed by the counter ID in the brw_oa_counter_id enum.
    *
    * XXX: We can possibly get rid of this if we don't can any
    * exceptional condition for triggering a partial accumulation of
    * results. (previously we would accumulate if we ran out of space
    * in bookend_bo) We could hit problems with overflowing the perf
    * circular buffer, but maybe for those cases we can instead simply
    * report an error with the counters?
    */
   uint64_t oa_accumulator[MAX_OA_COUNTERS];

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

/* Samples read from the perf circular buffer */
struct oa_perf_sample {
   struct perf_event_header header;
   uint64_t time; /* PERF_SAMPLE_TIME */
   uint64_t value; /* PERF_SAMPLE_READ */
   uint32_t raw_size;
   uint8_t raw_data[];
};
#define MAX_OA_PERF_SAMPLE_SIZE (8 +   /* perf_event_header */       \
                                 8 +   /* time: TODO remove */       \
                                 8 +   /* value: TODO remove */      \
                                 4 +   /* raw_size */                \
                                 256 + /* raw OA counter snapshot */ \
                                 4)    /* alignment padding */

#define TAKEN(HEAD, TAIL, POT_SIZE)	(((HEAD) - (TAIL)) & (POT_SIZE - 1))

/* Note: this will equate to 0 when the buffer is exactly full... */
#define REMAINING(HEAD, TAIL, POT_SIZE) (POT_SIZE - TAKEN (HEAD, TAIL, POT_SIZE))

#if defined(__i386__)
#define rmb()           __asm__ volatile("lock; addl $0,0(%%esp)" ::: "memory")
#define mb()            __asm__ volatile("lock; addl $0,0(%%esp)" ::: "memory")
#endif

#if defined(__x86_64__)
#define rmb()           __asm__ volatile("lfence" ::: "memory")
#define mb()            __asm__ volatile("mfence" ::: "memory")
#endif

/* TODO: consider using <stdatomic.h> something like:
 *
 * #define rmb() atomic_thread_fence(memory_order_seq_consume)
 * #define mb() atomic_thread_fence(memory_order_seq_cst)
 */


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

   DBG("%4d  %-7s %-6s %-10s %-11s %-6s %-9s\n",
       name,
       m->Active ? "Active" : "",
       m->Ended ? "Ended" : "",
       resolved,
       brw_is_perf_monitor_result_available(ctx, m) ? "Available" : "",
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

   if (brw->gen == 7) {
      BEGIN_BATCH(3);
      OUT_BATCH(GEN6_MI_REPORT_PERF_COUNT);
      OUT_RELOC(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else {
      unreachable("Unsupported generation for OA performance counters.");
   }

   /* Reports apparently don't always get written unless we flush after. */
   intel_batchbuffer_emit_mi_flush(brw);

   (void) batch_used;
   assert(brw->batch.used - batch_used <= MI_REPORT_PERF_COUNT_BATCH_DWORDS * 4);
}

static unsigned int
read_perf_head(struct perf_event_mmap_page *mmap_page)
{
   unsigned int head = (*(volatile uint64_t *)&mmap_page->data_head);
   rmb();

   return head;
}

static void
write_perf_tail(struct perf_event_mmap_page *mmap_page,
                unsigned int tail)
{
   /* Make sure we've finished reading all the sample data we
    * we're consuming before updating the tail... */
   mb();
   mmap_page->data_tail = tail;
}

/* Update the real perf tail pointer according to the monitor tail that
 * is currently furthest behind...
 */
static void
update_perf_tail(struct brw_context *brw)
{
   unsigned int size = brw->perfmon.perf_oa_buffer_size;
   unsigned int head = read_perf_head(brw->perfmon.perf_oa_mmap_page);
   int straggler_taken = -1;
   unsigned int straggler_tail;

   for (int i = 0; i < brw->perfmon.unresolved_elements; i++) {
      struct brw_perf_monitor_object *monitor = brw->perfmon.unresolved[i];
      int taken = TAKEN(head, monitor->oa_tail, size);

      if (taken > straggler_taken) {
         straggler_taken = taken;
         straggler_tail = monitor->oa_tail;
      }
   }

   if (straggler_taken >= 0)
      write_perf_tail(brw->perfmon.perf_oa_mmap_page, straggler_tail);
}

/**
 * Add a monitor to the global list of "unresolved monitors."
 *
 * Monitors are "unresolved" until all the counter snapshots have been
 * accumulated via accumulate_oa_snapshots() after the end MI_REPORT_PERF_COUNT
 * has landed in monitor->oa_bo.
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

   update_perf_tail(brw);
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
         break;
      }
   }

   update_perf_tail(brw);
}

/* XXX: For BDW+ we'll need to check fuse registers */
static int
get_eu_count(uint32_t devid)
{
   const struct brw_device_info *info = brw_get_device_info(devid);

   assert(info && info->is_haswell);

   if (info->gt == 1)
      return 10;
   else if (info->gt == 2)
      return 20;
   else if (info->gt == 3)
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
}

/**
 * Accumulate OA counter results from a series of snapshots.
 *
 * N.B. We write snapshots for the beginning and end of a monitor into
 * monitor->oa_bo as well as collect periodic snapshots from the Linux
 * perf interface.
 *
 * These periodic snapshots help to ensure we handle counter overflow
 * correctly by being frequent enough to ensure we don't miss multiple
 * wrap overflows of a counter between snapshots.
 */
static void
accumulate_oa_snapshots(struct brw_context *brw,
                        struct brw_perf_monitor_object *monitor)
{
   struct gl_perf_monitor_object *m = &monitor->base;
   uint32_t *monitor_buffer = monitor->oa_bo->virtual;
   uint8_t *data = brw->perfmon.perf_oa_mmap_base + brw->perfmon.page_size;
   const unsigned int size = brw->perfmon.perf_oa_buffer_size;
   const uint64_t mask = size - 1;
   uint64_t head;
   uint64_t tail;
   uint64_t dummy_count;
   uint32_t *start;
   uint64_t start_timestamp;
   uint32_t *last;
   uint32_t *end;
   uint64_t end_timestamp;
   uint8_t scratch[MAX_OA_PERF_SAMPLE_SIZE];

   assert(m->Ended);

   /* A well defined side effect of reading the sample count of
    * an i915 OA event is that all outstanding counter reports
    * will be flushed into the perf mmap buffer... */
   read(brw->perfmon.perf_oa_event_fd, &dummy_count, 8);

   start = last = monitor_buffer;
   end = monitor_buffer + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint32_t));

   /* XXX: Is there anything we can do to handle this gracefully/
    * report the error to the application? */
   if (start[0] != monitor->current_report_id)
      DBG("Spurious start report id=%"PRIu32"\n", start[0]);
   if (end[0] != (monitor->current_report_id + 1))
      DBG("Spurious end report id=%"PRIu32"\n", start[0]);

   start_timestamp = read_report_timestamp(brw, start);
   end_timestamp = read_report_timestamp(brw, end);

   head = read_perf_head(brw->perfmon.perf_oa_mmap_page);
   tail = monitor->oa_tail;

   //fprintf(stderr, "Handle event mask = 0x%" PRIx64
   //        " head=%" PRIu64 " tail=%" PRIu64 "\n", mask, head, tail);

   while (TAKEN(head, tail, size)) {
      const struct perf_event_header *header =
         (const struct perf_event_header *)(data + (tail & mask));

      if (header->size == 0) {
         DBG("Spurious header size == 0\n");
         /* XXX: How should we handle this instead of exiting() */
         exit(1);
      }

      if (header->size > (head - tail)) {
         DBG("Spurious header size would overshoot head\n");
         /* XXX: How should we handle this instead of exiting() */
         exit(1);
      }

      //fprintf(stderr, "header = %p tail=%" PRIu64 " size=%d\n",
      //        header, tail, header->size);

      if ((const uint8_t *)header + header->size > data + size) {
         int before;

         if (header->size > MAX_OA_PERF_SAMPLE_SIZE) {
            DBG("Skipping spurious sample larger than expected\n");
            tail += header->size;
            continue;
         }

         before = data + size - (const uint8_t *)header;

         memcpy(scratch, header, before);
         memcpy(scratch + before, data, header->size - before);

         header = (struct perf_event_header *)scratch;
         //fprintf(stderr, "DEBUG: split\n");
         //exit(1);
      }

      switch (header->type) {
         case PERF_RECORD_LOST: {
            struct {
               struct perf_event_header header;
               uint64_t id;
               uint64_t n_lost;
            } *lost = (void *)header;
            DBG("i915_oa: Lost %" PRIu64 " events\n", lost->n_lost);
            break;
         }

         case PERF_RECORD_THROTTLE:
            DBG("i915_oa: Sampling has been throttled\n");
            break;

         case PERF_RECORD_UNTHROTTLE:
            DBG("i915_oa: Sampling has been unthrottled\n");
            break;

         case PERF_RECORD_SAMPLE: {
            struct oa_perf_sample *perf_sample = (struct oa_perf_sample *)header;
            uint32_t *report = (uint32_t *)perf_sample->raw_data;
            uint64_t timestamp = read_report_timestamp(brw, report);

            if (timestamp >= end_timestamp)
               goto end;

            if (timestamp > start_timestamp) {
               add_deltas(brw, monitor, last, report);
               last = report;
            }

            break;
         }

         default:
            DBG("i915_oa: Spurious header type = %d\n", header->type);
      }

      //fprintf(stderr, "Tail += %d\n", header->size);

      tail += header->size;
   }

end:

   add_deltas(brw, monitor, last, end);

   DBG("Marking %d resolved - results gathered\n", m->Name);
   monitor->resolved = true;
   drop_from_unresolved_monitor_list(brw, monitor);
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

static bool
open_i915_oa_event (struct brw_context *brw,
                    uint64_t report_format,
                    int period_exponent,
                    int drm_fd,
                    uint32_t ctx_id)
{
   struct perf_event_attr attr;
   int event_fd;
   void *mmap_base;

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

   event_fd = perf_event_open(&attr,
                              -1,  /* pid */
                              0, /* cpu */
                              -1, /* group fd */
                              PERF_FLAG_FD_CLOEXEC); /* flags */
   if (event_fd == -1) {
      DBG("Error opening i915_oa perf event: %m\n");
      return false;
   }

   /* NB: A read-write mapping ensures the kernel will stop writing data when
    * the buffer is full, and will report samples as lost. */
   mmap_base = mmap(NULL,
                    brw->perfmon.perf_oa_buffer_size + brw->perfmon.page_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, event_fd, 0);
   if (mmap_base == MAP_FAILED) {
      DBG("Error mapping circular buffer, %m\n");
      close (event_fd);
      return false;
   }

   brw->perfmon.perf_oa_event_fd = event_fd;
   brw->perfmon.perf_oa_mmap_base = mmap_base;
   brw->perfmon.perf_oa_mmap_page = mmap_base;

   return true;
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
   monitor->current_report_id = brw->perfmon.next_query_start_report_id;
   brw->perfmon.next_query_start_report_id += 2;

   drop_from_unresolved_monitor_list(brw, monitor);
   monitor->resolved = false;

   memset(monitor->oa_accumulator, 0, sizeof(monitor->oa_accumulator));

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

         /* The timestamp for HSW+ increments every 80ns
          *
          * The period_exponent gives a sampling period as follows:
          *   sample_period = 80ns * 2^(period_exponent + 1)
          *
          * The overflow period for Haswell can be calculated as:
          *
          * 2^32 / (n_eus * max_gen_freq * 2)
          * (E.g. 40 EUs @ 1GHz = ~53ms)
          *
          * We currently sample every 42 milliseconds...
          */
         period_exponent = 18;

         if (!open_i915_oa_event(brw,
                                 I915_PERF_OA_FORMAT_A45_B8_C8_HSW,
                                 period_exponent,
                                 screen->fd, /* drm fd */
                                 brw->hw_ctx->ctx_id))
            return GL_FALSE; /* XXX: do we need to set GL error state? */
      }

      if (brw->perfmon.oa_users == 0 &&
          ioctl(brw->perfmon.perf_oa_event_fd, PERF_EVENT_IOC_ENABLE, 0) < 0)
      {
         DBG("WARNING: Error enabling i915_oa perf event: %m\n");
         return GL_FALSE; /* XXX: do we need to set GL error state? */
      }

      monitor->oa_bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. monitor OA bo", 4096, 64);
#ifdef DEBUG
      /* Pre-filling the BO helps debug whether writes landed. */
      drm_intel_bo_map(monitor->oa_bo, true);
      memset((char *) monitor->oa_bo->virtual, 0x80, 4096);
      drm_intel_bo_unmap(monitor->oa_bo);
#endif

      /* Take a starting OA counter snapshot. */
      emit_mi_report_perf_count(brw, monitor->oa_bo, 0,
                                monitor->current_report_id);

      /* Each unresolved monitor maintains a separate tail pointer into the
       * circular perf sample buffer. The real tail pointer in
       * perfmon.perf_oa_mmap_page.data_tail will correspond to the monitor
       * tail that is furthest behind.
       */
      monitor->oa_tail = read_perf_head(brw->perfmon.perf_oa_mmap_page);

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
                                SECOND_SNAPSHOT_OFFSET_IN_BYTES,
                                monitor->current_report_id + 1);

      --brw->perfmon.open_oa_monitors;

      /* NB: even though the monitor has now ended, it can't be resolved
       * until the end MI_REPORT_PERF_COUNT snapshot has been written
       * to monitor->oa_bo */
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

   if (monitor_needs_oa(brw, m)) {
      int n_oa_counters = ctx->PerfMonitor.Groups[OA_COUNTERS].NumCounters;

      drm_intel_bo_map(monitor->oa_bo, false);

      accumulate_oa_snapshots(brw, monitor);
      assert(monitor->resolved);

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

/* TODO: we can remove both of these since we now collect intermediate
 * OA counter snapshots for detecting counter overflow via the Linux
 * perf interface instead...
 */

/**
 * Called at the start of every render ring batch.
 */
void
brw_perf_monitor_new_batch(struct brw_context *brw)
{
   /* TODO: remove this unused hook */
}

/**
 * Called at the end of every render ring batch.
 */
void
brw_perf_monitor_finish_batch(struct brw_context *brw)
{
   /* TODO: remove this unused hook */
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

   brw->perfmon.page_size = sysconf(_SC_PAGE_SIZE);

   brw->perfmon.perf_oa_event_fd = -1;
   brw->perfmon.perf_oa_buffer_size = 1024 * 1024; /* NB: must be power of two */

   memset(brw->perfmon.oa_counters, 0, sizeof(brw->perfmon.oa_counters));
   init_hsw_oa_counters(brw);

   brw->perfmon.next_query_start_report_id = 1000;

   eu_count = get_eu_count(brw->intelScreen->deviceID);
}

void
brw_destroy_performance_monitors(struct brw_context *brw)
{
   if (brw->perfmon.perf_oa_event_fd != -1) {
      if (brw->perfmon.perf_oa_mmap_base) {
         size_t mapping_len =
            brw->perfmon.perf_oa_buffer_size + brw->perfmon.page_size;

         munmap(brw->perfmon.perf_oa_mmap_base, mapping_len);
         brw->perfmon.perf_oa_mmap_base = NULL;
      }

      close(brw->perfmon.perf_oa_event_fd);
      brw->perfmon.perf_oa_event_fd = -1;
   }
}
