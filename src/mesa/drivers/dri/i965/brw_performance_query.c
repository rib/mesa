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
 * \file brw_performance_query.c
 *
 * Implementation of the GL_INTEL_performance_query extension.
 *
 * Currently there are two possible counter sources exposed here:
 *
 * On Gen6+ hardware we have numerous 64bit Pipeline Statistics Registers
 * that we can snapshot at the beginning and end of a query.
 *
 * On Gen7.5+ we have Observability Architecture counters which are
 * covered in separate document from the rest of the PRMs.  It is available at:
 * https://01.org/linuxgraphics/documentation/driver-documentation-prms
 * => 2013 Intel Core Processor Family => Observability Performance Counters
 * (This one volume covers Sandybridge, Ivybridge, Baytrail, and Haswell,
 * though notably we currently only support OA counters for Haswell+)
 */

#include <linux/perf_event.h>

#include <limits.h>

#include <asm/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <xf86drm.h>

#include "main/hash.h"
#include "main/macros.h"
#include "main/mtypes.h"
#include "main/performance_query.h"

#include "util/bitset.h"
#include "util/ralloc.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_performance_query.h"
#include "brw_oa_hsw.h"
#include "brw_oa_bdw.h"
#include "intel_batchbuffer.h"

#define FILE_DEBUG_FLAG DEBUG_PERFMON

#define MAX_OA_REPORT_COUNTERS 62

#define OAREPORT_REASON_MASK           0x3f
#define OAREPORT_REASON_SHIFT          19
#define OAREPORT_REASON_TIMER          (1<<0)
#define OAREPORT_REASON_TRIGGER1       (1<<1)
#define OAREPORT_REASON_TRIGGER2       (1<<2)
#define OAREPORT_REASON_CTX_SWITCH     (1<<3)
#define OAREPORT_REASON_GO_TRANSITION  (1<<4)


/**
 * i965 representation of a performance query object.
 *
 * NB: We want to keep this structure relatively lean considering that
 * applications may expect to allocate enough objects to be able to
 * query around all draw calls in a frame.
 */
struct brw_perf_query_object
{
   /** The base class. */
   struct gl_perf_query_object base;

   const struct brw_perf_query *query;

   /* See query->kind to know which state below is in use... */
   union {
      struct {

         /**
          * BO containing OA counter snapshots at query Begin/End time.
          */
         drm_intel_bo *bo;
         int current_report_id;

         /**
          * We collect periodic counter snapshots via perf so we can account
          * for counter overflow and this is a pointer into the circular
          * perf buffer for collecting snapshots that lie within the begin-end
          * bounds of this query.
          */
         unsigned int perf_tail;

         /**
          * Storage the final accumulated OA counters.
          */
         uint64_t accumulator[MAX_OA_REPORT_COUNTERS];

         /**
          * false while in the unresolved_elements list, and set to true when
          * the final, end MI_RPC snapshot has been accumulated.
          */
         bool results_accumulated;

      } oa;

      struct {
         /**
          * BO containing starting and ending snapshots for the
          * statistics counters.
          */
         drm_intel_bo *bo;

         /**
          * Storage for final pipeline statistics counter results.
          */
         uint64_t *results;

      } pipeline_stats;
   };
};

/* Samples read from the perf circular buffer */
struct oa_perf_sample {
   struct perf_event_header header;
   uint32_t raw_size;
   uint8_t raw_data[];
};
#define MAX_OA_PERF_SAMPLE_SIZE (8 +   /* perf_event_header */       \
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

/* Allow building for a more recent kernel than the system headers
 * correspond too... */
#ifndef PERF_RECORD_DEVICE
#define PERF_RECORD_DEVICE                   13
#endif

/** Downcasting convenience macro. */
static inline struct brw_perf_query_object *
brw_perf_query(struct gl_perf_query_object *o)
{
   return (struct brw_perf_query_object *) o;
}

#define SECOND_SNAPSHOT_OFFSET_IN_BYTES 2048

/******************************************************************************/

static GLboolean brw_is_perf_query_ready(struct gl_context *,
					 struct gl_perf_query_object *);

static void
dump_perf_query_callback(GLuint id, void *query_void, void *brw_void)
{
   struct gl_context *ctx = brw_void;
   struct gl_perf_query_object *o = query_void;
   struct brw_perf_query_object *obj = query_void;

   switch(obj->query->kind) {
   case OA_COUNTERS:
      DBG("%4d: %-6s %-8s BO: %-4s OA data: %-10s %-15s\n",
          id,
          o->Used ? "Dirty," : "New,",
          o->Active ? "Active," : (o->Ready ? "Ready," : "Pending,"),
          obj->oa.bo ? "yes," : "no,",
          brw_is_perf_query_ready(ctx, o) ? "ready," : "not ready,",
          obj->oa.results_accumulated ? "accumulated" : "not accumulated");
      break;
   case PIPELINE_STATS:
      DBG("%4d: %-6s %-8s BO: %-4s\n",
          id,
          o->Used ? "Dirty," : "New,",
          o->Active ? "Active," : (o->Ready ? "Ready," : "Pending,"),
          obj->pipeline_stats.bo ? "yes" : "no");
      break;
   }
}

void
brw_dump_perf_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;
   DBG("Queries: (Open queries = %d, OA users = %d)\n",
       brw->perfquery.n_active_oa_queries, brw->perfquery.n_oa_users);
   _mesa_HashWalk(ctx->PerfQuery.Objects, dump_perf_query_callback, brw);
}

/******************************************************************************/

static void
brw_get_perf_query_info(struct gl_context *ctx,
                        int query_index,
                        const char **name,
                        GLuint *data_size,
                        GLuint *n_counters,
                        GLuint *n_active)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];

   *name = query->name;
   *data_size = query->data_size;
   *n_counters = query->n_counters;

   switch(query->kind) {
   case OA_COUNTERS:
      *n_active = brw->perfquery.n_active_oa_queries;
      break;

   case PIPELINE_STATS:
      *n_active = brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }
}

static void
brw_get_perf_counter_info(struct gl_context *ctx,
                          int query_index,
                          int counter_index,
                          const char **name,
                          const char **desc,
                          GLuint *offset,
                          GLuint *data_size,
                          GLuint *type_enum,
                          GLuint *data_type_enum,
                          GLuint64 *raw_max)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];
   const struct brw_perf_query_counter *counter =
      &query->counters[counter_index];

   *name = counter->name;
   *desc = counter->desc;
   *offset = counter->offset;
   *data_size = counter->size;
   *type_enum = counter->type;
   *data_type_enum = counter->data_type;
   *raw_max = counter->raw_max;
}

/**
 * Take a snapshot of any queried pipeline statistics counters.
 */
static void
snapshot_statistics_registers(struct brw_context *brw,
                              struct brw_perf_query_object *obj,
                              uint32_t offset_in_bytes)
{
   const int offset = offset_in_bytes / sizeof(uint64_t);
   const struct brw_perf_query *query = obj->query;
   const int n_counters = query->n_counters;

   intel_batchbuffer_emit_mi_flush(brw);

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];

      assert(counter->data_type == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);

      brw_store_register_mem64(brw, obj->pipeline_stats.bo,
                               counter->pipeline_stat.reg,
                               offset + i);
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
                          struct brw_perf_query_object *obj)
{
   const struct brw_perf_query *query = obj->query;
   const int n_counters = query->n_counters;

   obj->pipeline_stats.results = calloc(n_counters, sizeof(uint64_t));
   if (obj->pipeline_stats.results == NULL) {
      _mesa_error_no_memory(__func__);
      return;
   }

   drm_intel_bo_map(obj->pipeline_stats.bo, false);
   uint64_t *start = obj->pipeline_stats.bo->virtual;
   uint64_t *end = start + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint64_t));

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];
      obj->pipeline_stats.results[i] = end[i] - start[i];

      if (counter->pipeline_stat.numerator !=
          counter->pipeline_stat.denominator) {
         obj->pipeline_stats.results[i] *= counter->pipeline_stat.numerator;
         obj->pipeline_stats.results[i] /= counter->pipeline_stat.denominator;
      }
   }

   drm_intel_bo_unmap(obj->pipeline_stats.bo);
   drm_intel_bo_unreference(obj->pipeline_stats.bo);
   obj->pipeline_stats.bo = NULL;
}

/******************************************************************************/

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

   /* Reports apparently don't always get written unless we flush first. */
   intel_batchbuffer_emit_mi_flush(brw);

   if (brw->gen < 8) {
      BEGIN_BATCH(3);
      OUT_BATCH(GEN6_MI_REPORT_PERF_COUNT);
      OUT_RELOC(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else {
      BEGIN_BATCH(4);
      OUT_BATCH(GEN8_MI_REPORT_PERF_COUNT);
      OUT_RELOC64(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                  offset_in_bytes);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   }

   /* Reports apparently don't always get written unless we flush after. */
   intel_batchbuffer_emit_mi_flush(brw);
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

/* Update the real perf tail pointer according to the query tail that
 * is currently furthest behind...
 */
static void
update_perf_tail(struct brw_context *brw)
{
   unsigned int size = brw->perfquery.perf_oa_buffer_size;
   unsigned int head = read_perf_head(brw->perfquery.perf_oa_mmap_page);
   int straggler_taken = -1;
   unsigned int straggler_tail;

   for (int i = 0; i < brw->perfquery.unresolved_elements; i++) {
      struct brw_perf_query_object *obj = brw->perfquery.unresolved[i];
      int taken;

      if (!obj->oa.bo)
         continue;

      taken = TAKEN(head, obj->oa.perf_tail, size);

      if (taken > straggler_taken) {
         straggler_taken = taken;
         straggler_tail = obj->oa.perf_tail;
      }
   }

   if (straggler_taken >= 0)
      write_perf_tail(brw->perfquery.perf_oa_mmap_page, straggler_tail);
}

/**
 * Add a query to the global list of "unresolved queries."
 *
 * Queries are "unresolved" until all the counter snapshots have been
 * accumulated via accumulate_oa_snapshots() after the end MI_REPORT_PERF_COUNT
 * has landed in query->oa.bo.
 */
static void
add_to_unresolved_query_list(struct brw_context *brw,
                             struct brw_perf_query_object *obj)
{
   if (brw->perfquery.unresolved_elements >=
       brw->perfquery.unresolved_array_size) {
      brw->perfquery.unresolved_array_size *= 1.5;
      brw->perfquery.unresolved = reralloc(brw, brw->perfquery.unresolved,
                                           struct brw_perf_query_object *,
                                           brw->perfquery.unresolved_array_size);
   }

   brw->perfquery.unresolved[brw->perfquery.unresolved_elements++] = obj;

   if (obj->oa.bo)
      update_perf_tail(brw);
}

/**
 * Remove a query from the global list of "unresolved queries." once
 * the end MI_RPC OA counter snapshot has been accumulated, or when
 * discarding unwanted query results.
 */
static void
drop_from_unresolved_query_list(struct brw_context *brw,
                                struct brw_perf_query_object *obj)
{
   for (int i = 0; i < brw->perfquery.unresolved_elements; i++) {
      if (brw->perfquery.unresolved[i] == obj) {
         int last_elt = --brw->perfquery.unresolved_elements;

         if (i == last_elt)
            brw->perfquery.unresolved[i] = NULL;
         else
            brw->perfquery.unresolved[i] = brw->perfquery.unresolved[last_elt];

         break;
      }
   }

   if (obj->oa.bo)
      update_perf_tail(brw);
}

/* XXX: should we add these directly to brw_device_info maybe? */
static void
init_dev_info(struct brw_context *brw)
{
   const struct brw_device_info *info = brw->intelScreen->devinfo;
   __DRIscreen *screen = brw->intelScreen->driScrnPriv;

   if (brw->is_haswell) {
      if (info->gt == 1) {
         brw->perfquery.devinfo.n_eus = 10;
         brw->perfquery.devinfo.n_eu_slices = 1;
      } else if (info->gt == 2) {
         brw->perfquery.devinfo.n_eus = 20;
         brw->perfquery.devinfo.n_eu_slices = 1;
      } else if (info->gt == 3) {
         brw->perfquery.devinfo.n_eus = 40;
         brw->perfquery.devinfo.n_eu_slices = 2;
      }
   } else {
#ifdef I915_PARAM_EU_TOTAL
      drm_i915_getparam_t gp;
      int ret;
      int n_eus;
      int slice_mask;

      gp.param = I915_PARAM_EU_TOTAL;
      gp.value = &n_eus;
      ret = drmIoctl(screen->fd, DRM_IOCTL_I915_GETPARAM, &gp);
      assert(ret == 0 && n_eus > 0);

      gp.param = I915_PARAM_SLICE_MASK;
      gp.value = &slice_mask;
      ret = drmIoctl(screen->fd, DRM_IOCTL_I915_GETPARAM, &gp);
      assert(ret == 0 && slice_mask);

      brw->perfquery.devinfo.n_eus = n_eus;
      brw->perfquery.devinfo.n_eu_slices = _mesa_bitcount(slice_mask);
#else
      assert(0);
#endif
   }
}

static uint64_t
read_report_timestamp(uint32_t *report)
{
   /* The least significant timestamp bit represents 80ns */
   return ((uint64_t)report[1]) * 80;
}

static void
accumulate_uint32(const uint32_t *report0,
                  const uint32_t *report1,
                  uint64_t *accumulator)
{
   *accumulator += (uint32_t)(*report1 - *report0);
}

static void
accumulate_uint40(int a_index,
                  const uint32_t *report0,
                  const uint32_t *report1,
                  uint64_t *accumulator)
{
   const uint8_t *high_bytes0 = (uint8_t *)(report0 + 40);
   const uint8_t *high_bytes1 = (uint8_t *)(report1 + 40);
   uint64_t high0 = (uint64_t)(high_bytes0[a_index]) << 32;
   uint64_t high1 = (uint64_t)(high_bytes1[a_index]) << 32;
   uint64_t value0 = report0[a_index + 4] | high0;
   uint64_t value1 = report1[a_index + 4] | high1;
   uint64_t delta;

   if (value0 > value1)
      delta = (1ULL << 40) + value1 - value0;
   else
      delta = value1 - value0;

   *accumulator += delta;
}

/**
 * Given pointers to starting and ending OA snapshots, add the deltas for each
 * counter to the results.
 */
static void
add_deltas(struct brw_context *brw,
           struct brw_perf_query_object *obj,
           const uint32_t *start,
           const uint32_t *end)
{
   const struct brw_perf_query *query = obj->query;
   uint64_t *accumulator = obj->oa.accumulator;
   int idx = 0;
   int i;

   switch (query->oa_format) {
   case I915_OA_FORMAT_A32u40_A4u32_B8_C8:
      accumulate_uint32(start + 1, end + 1, accumulator + idx++); /* timestamp */
      accumulate_uint32(start + 3, end + 3, accumulator + idx++); /* clock */

      /* 32x 40bit A counters... */
      for (i = 0; i < 32; i++)
         accumulate_uint40(i, start, end, accumulator + idx++);

      /* 4x 32bit A counters... */
      for (i = 0; i < 4; i++)
         accumulate_uint32(start + 36 + i, end + 36 + i, accumulator + idx++);

      /* 8x 32bit B counters + 8x 32bit C counters... */
      for (i = 0; i < 16; i++)
         accumulate_uint32(start + 48 + i, end + 48 + i, accumulator + idx++);

      break;
   case I915_OA_FORMAT_A45_B8_C8:
      accumulate_uint32(start + 1, end + 1, accumulator); /* timestamp */

      for (i = 0; i < 61; i++)
         accumulate_uint32(start + 3 + i, end + 3 + i, accumulator + 1 + i);

      break;
   default:
      unreachable("Can't accumulate OA counters in unknown format");
   }
}

/* Handle restarting ioctl if interrupted... */
static int
perf_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   return ret;
}

static bool
inc_n_oa_users(struct brw_context *brw)
{
   if (brw->perfquery.n_oa_users == 0 &&
       perf_ioctl(brw->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_ENABLE, 0) < 0)
   {
      return false;
   }
   ++brw->perfquery.n_oa_users;

   return true;
}

static void
dec_n_oa_users(struct brw_context *brw)
{
   /* Disabling the i915_oa event will effectively disable the OA
    * counters.  Note it's important to be sure there are no outstanding
    * MI_RPC commands at this point since they could stall the CS
    * indefinitely once OACONTROL is disabled.
    */
   --brw->perfquery.n_oa_users;
   if (brw->perfquery.n_oa_users == 0 &&
       perf_ioctl(brw->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_DISABLE, 0) < 0)
   {
      DBG("WARNING: Error disabling i915_oa perf event: %m\n");
   }
}

/* In general if we see anything spurious while accumulating results,
 * we don't try and continue accumulating the current query, hoping
 * for the best, we scrap anything outstanding, and then hope for the
 * best with new queries. */
static void
discard_all_queries(struct brw_context *brw)
{
   while (brw->perfquery.unresolved_elements) {
      struct brw_perf_query_object *obj = brw->perfquery.unresolved[0];

      obj->oa.results_accumulated = true;
      drop_from_unresolved_query_list(brw, brw->perfquery.unresolved[0]);

      dec_n_oa_users(brw);
   }
}

/**
 * Accumulate OA counter results from a series of snapshots.
 *
 * N.B. We write snapshots for the beginning and end of a query into
 * query->oa.bo as well as collect periodic snapshots from the Linux
 * perf interface.
 *
 * These periodic snapshots help to ensure we handle counter overflow
 * correctly by being frequent enough to ensure we don't miss multiple
 * overflows of a counter between snapshots.
 */
static void
accumulate_oa_snapshots(struct brw_context *brw,
                        struct brw_perf_query_object *obj)
{
   struct gl_perf_query_object *o = &obj->base;
   uint32_t *query_buffer;
   uint8_t *data = brw->perfquery.perf_oa_mmap_base + brw->perfquery.page_size;
   const unsigned int size = brw->perfquery.perf_oa_buffer_size;
   const uint64_t mask = size - 1;
   uint64_t head;
   uint64_t tail;
   uint32_t *start;
   uint64_t start_timestamp;
   uint32_t *last;
   uint32_t *end;
   uint64_t end_timestamp;
   uint8_t scratch[MAX_OA_PERF_SAMPLE_SIZE];

   assert(o->Ready);

   if (fsync(brw->perfquery.perf_oa_event_fd)  < 0)
      DBG("Failed to flush outstanding perf events: %m\n");

   drm_intel_bo_map(obj->oa.bo, false);
   query_buffer = obj->oa.bo->virtual;

   start = last = query_buffer;
   end = query_buffer + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint32_t));

   if (start[0] != obj->oa.current_report_id) {
      DBG("Spurious start report id=%"PRIu32"\n", start[0]);
      goto error;
   }
   if (end[0] != (obj->oa.current_report_id + 1)) {
      DBG("Spurious end report id=%"PRIu32"\n", end[0]);
      goto error;
   }

   start_timestamp = read_report_timestamp(start);
   end_timestamp = read_report_timestamp(end);

   head = read_perf_head(brw->perfquery.perf_oa_mmap_page);
   tail = obj->oa.perf_tail;

   while (TAKEN(head, tail, size)) {
      const struct perf_event_header *header =
         (const struct perf_event_header *)(data + (tail & mask));

      assert(header->size != 0);
      assert(header->size <= (head - tail));

      if ((const uint8_t *)header + header->size > data + size) {
         int before;

         if (header->size > sizeof(scratch)) {
            DBG("Spurious sample larger than expected\n");
            goto error;
         }

         before = data + size - (const uint8_t *)header;

         memcpy(scratch, header, before);
         memcpy(scratch + before, data, header->size - before);

         header = (struct perf_event_header *)scratch;
      }

      switch (header->type) {
         case PERF_RECORD_LOST: {
            struct {
               struct perf_event_header header;
               uint64_t id;
               uint64_t n_lost;
            } *lost = (void *)header;

            DBG("i915_oa: Lost %" PRIu64 " events\n", lost->n_lost);
            goto error;
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
            uint64_t timestamp = read_report_timestamp(report);

            if (timestamp >= end_timestamp)
               goto end;

            if (timestamp > start_timestamp) {
               uint32_t reason = (report[0] >> OAREPORT_REASON_SHIFT) &
                  OAREPORT_REASON_MASK;

               /* Since the counters continue while other contexts are
                * running we need to discount any unrelated delta. The
                * hardware automatically generates a report on context
                * switch which gives us a new reference point to
                * continuing adding deltas from.
                */
               if (!(reason & OAREPORT_REASON_CTX_SWITCH))
                  add_deltas(brw, obj, last, report);

               last = report;
            }

            break;
         }

         case PERF_RECORD_DEVICE: {
	    struct i915_oa_event {
		struct perf_event_header header;
		drm_i915_oa_event_header_t oa_header;
	    } *oa_event = (void *)header;

	    switch (oa_event->oa_header.type) {
	    case I915_OA_RECORD_BUFFER_OVERFLOW:
		DBG("i915_oa: OA buffer overflow\n");
		break;
	    case I915_OA_RECORD_REPORT_LOST:
		DBG("i915_oa: OA report lost\n");
		break;
	    }

	    break;
         }

         default:
            DBG("i915_oa: Spurious header type = %d\n", header->type);
      }

      tail += header->size;
   }

end:

   add_deltas(brw, obj, last, end);

   DBG("Marking %d resolved - results gathered\n", o->Id);

   drm_intel_bo_unmap(obj->oa.bo);
   obj->oa.results_accumulated = true;
   drop_from_unresolved_query_list(brw, obj);
   dec_n_oa_users(brw);

   return;

error:

   drm_intel_bo_unmap(obj->oa.bo);
   discard_all_queries(brw);
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
   return read_file_uint64("/sys/bus/event_source/devices/i915_oa/type");
}

static long
perf_event_open (struct perf_event_attr *hw_event,
                 pid_t pid,
                 int cpu,
                 int group_fd,
                 unsigned long flags)
{
   return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static bool
open_i915_oa_event(struct brw_context *brw,
                   int metrics_set,
                   int report_format,
                   int period_exponent,
                   int drm_fd,
                   uint32_t ctx_id)
{
   struct perf_event_attr attr;
   drm_i915_oa_attr_t oa_attr;
   int event_fd;
   void *mmap_base;

   memset(&attr, 0, sizeof(attr));
   attr.size = sizeof(attr);
   attr.type = lookup_i915_oa_id();

   attr.sample_type = PERF_SAMPLE_RAW;
   attr.disabled = 1;
   attr.sample_period = 1;

   memset(&oa_attr, 0, sizeof(oa_attr));
   oa_attr.size = sizeof(oa_attr);

   oa_attr.format = report_format;
   oa_attr.metrics_set = metrics_set;
   oa_attr.timer_exponent = period_exponent;

   oa_attr.single_context = true;
   oa_attr.ctx_id = ctx_id;
   oa_attr.drm_fd = drm_fd;

   attr.config = (uint64_t)&oa_attr;

   event_fd = perf_event_open(&attr,
                              -1, /* pid */
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
                    brw->perfquery.perf_oa_buffer_size + brw->perfquery.page_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, event_fd, 0);
   if (mmap_base == MAP_FAILED) {
      DBG("Error mapping circular buffer, %m\n");
      close (event_fd);
      return false;
   }

   brw->perfquery.perf_oa_event_fd = event_fd;
   brw->perfquery.perf_oa_mmap_base = mmap_base;
   brw->perfquery.perf_oa_mmap_page = mmap_base;

   brw->perfquery.perf_oa_metrics_set = metrics_set;
   brw->perfquery.perf_oa_format = report_format;

   return true;
}

/**
 * Driver hook for glBeginPerfQueryINTEL().
 */
static GLboolean
brw_begin_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   const struct brw_perf_query *query = obj->query;

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Begin(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      /* If the OA counters aren't already on, enable them. */
      if (brw->perfquery.perf_oa_event_fd == -1) {
         __DRIscreen *screen = brw->intelScreen->driScrnPriv;
         uint32_t ctx_id = drm_intel_gem_context_get_context_id(brw->hw_ctx);
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
                                 query->oa_metrics_set,
                                 query->oa_format,
                                 period_exponent,
                                 screen->fd, /* drm fd */
                                 ctx_id))
            return false;
      } else {
         /* Opening an i915_oa event fd implies exclusive access to
          * the OA unit which will generate counter reports for a
          * specific counter set with a specific layout/format so we
          * can't begin any OA based queries that require a different
          * counter set or format unless we get an opportunity to
          * close the event fd and open a new one...
          */
         if (brw->perfquery.perf_oa_metrics_set != query->oa_metrics_set ||
             brw->perfquery.perf_oa_format != query->oa_format)
         {
            return false;
         }
      }

      if (!inc_n_oa_users(brw)) {
         DBG("WARNING: Error enabling i915_oa perf event: %m\n");
         return false;
      }

      if (obj->oa.bo) {
         drm_intel_bo_unreference(obj->oa.bo);
         obj->oa.bo = NULL;
      }

      obj->oa.bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. query OA bo", 4096, 64);
#ifdef DEBUG
      /* Pre-filling the BO helps debug whether writes landed. */
      drm_intel_bo_map(obj->oa.bo, true);
      memset((char *) obj->oa.bo->virtual, 0x80, 4096);
      drm_intel_bo_unmap(obj->oa.bo);
#endif

      obj->oa.current_report_id = brw->perfquery.next_query_start_report_id;
      brw->perfquery.next_query_start_report_id += 2;

      /* Take a starting OA counter snapshot. */
      emit_mi_report_perf_count(brw, obj->oa.bo, 0,
                                obj->oa.current_report_id);
      ++brw->perfquery.n_active_oa_queries;

      /* Each unresolved query maintains a separate tail pointer into the
       * circular perf sample buffer. The real tail pointer in
       * perfquery.perf_oa_mmap_page.data_tail will correspond to the query
       * tail that is furthest behind.
       */
      obj->oa.perf_tail = read_perf_head(brw->perfquery.perf_oa_mmap_page);

      memset(obj->oa.accumulator, 0, sizeof(obj->oa.accumulator));
      obj->oa.results_accumulated = false;

      add_to_unresolved_query_list(brw, obj);
      break;

   case PIPELINE_STATS:
      if (obj->pipeline_stats.bo) {
         drm_intel_bo_unreference(obj->pipeline_stats.bo);
         obj->pipeline_stats.bo = NULL;
      }

      obj->pipeline_stats.bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. query stats bo", 4096, 64);

      /* Take starting snapshots. */
      snapshot_statistics_registers(brw, obj, 0);

      free(obj->pipeline_stats.results);
      obj->pipeline_stats.results = NULL;

      ++brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }

   if (INTEL_DEBUG & DEBUG_PERFMON)
      brw_dump_perf_queries(brw);

   return true;
}

/**
 * Driver hook for glEndPerfQueryINTEL().
 */
static void
brw_end_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   DBG("End(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:

      /* NB: It's possible that the query will have already been marked
       * as 'accumulated' if an error was seen while reading samples
       * from perf. In this case we mustn't try and emit a closing
       * MI_RPC command in case the OA unit has already been disabled
       */
      if (!obj->oa.results_accumulated) {
         /* Take an ending OA counter snapshot. */
         emit_mi_report_perf_count(brw, obj->oa.bo,
                                   SECOND_SNAPSHOT_OFFSET_IN_BYTES,
                                   obj->oa.current_report_id + 1);
      }

      --brw->perfquery.n_active_oa_queries;

      /* NB: even though the query has now ended, it can't be resolved
       * until the end MI_REPORT_PERF_COUNT snapshot has been written
       * to query->oa.bo */
      break;

   case PIPELINE_STATS:
      /* Take ending snapshots. */
      snapshot_statistics_registers(brw, obj,
                                    SECOND_SNAPSHOT_OFFSET_IN_BYTES);
      --brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }
}

static void
brw_wait_perf_query(struct gl_context *ctx, struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   drm_intel_bo *bo = NULL;

   assert(!o->Ready);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      bo = obj->oa.bo;
      break;

   case PIPELINE_STATS:
      bo = obj->pipeline_stats.bo;
      break;
   }

   if (bo == NULL)
      return;

   /* If the current batch references our results bo then we need to
    * flush first... */
   if (drm_intel_bo_references(brw->batch.bo, bo))
      intel_batchbuffer_flush(brw);

   if (unlikely(brw->perf_debug)) {
      if (drm_intel_bo_busy(bo))
         perf_debug("Stalling GPU waiting for a performance query object.\n");
   }

   drm_intel_bo_wait_rendering(bo);
}

/**
 * Is a performance query result available?
 */
static GLboolean
brw_is_perf_query_ready(struct gl_context *ctx,
                        struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   if (o->Ready)
      return true;

   switch(obj->query->kind) {
   case OA_COUNTERS:
      return (obj->oa.results_accumulated ||
              (obj->oa.bo &&
               !drm_intel_bo_references(brw->batch.bo, obj->oa.bo) &&
               !drm_intel_bo_busy(obj->oa.bo)));

   case PIPELINE_STATS:
      return (obj->pipeline_stats.bo &&
              !drm_intel_bo_references(brw->batch.bo, obj->pipeline_stats.bo) &&
              !drm_intel_bo_busy(obj->pipeline_stats.bo));
   }

   unreachable("missing ready check for unknown query kind");
   return false;
}

static int
get_oa_counter_data(struct brw_context *brw,
                    struct brw_perf_query_object *obj,
                    size_t data_size,
                    uint8_t *data)
{
   const struct brw_perf_query *query = obj->query;
   int n_counters = query->n_counters;
   int written = 0;

   if (!obj->oa.results_accumulated) {
      accumulate_oa_snapshots(brw, obj);
      assert(obj->oa.results_accumulated);
   }

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];
      uint64_t *out_uint64;
      float *out_float;

      if (counter->size) {
         switch (counter->data_type) {
         case GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL:
            out_uint64 = (uint64_t *)(data + counter->offset);
            *out_uint64 = counter->oa_counter_read_uint64(brw, query,
                                                          obj->oa.accumulator);
            break;
         case GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL:
            out_float = (float *)(data + counter->offset);
            *out_float = counter->oa_counter_read_float(brw, query,
                                                        obj->oa.accumulator);
            break;
         default:
            /* So far we aren't using uint32, double or bool32... */
            unreachable("unexpected counter data type");
         }
         written = counter->offset + counter->size;
      }
   }

   return written;
}

static int
get_pipeline_stats_data(struct brw_context *brw,
                        struct brw_perf_query_object *obj,
                        size_t data_size,
                        uint8_t *data)

{
   int n_counters = obj->query->n_counters;
   uint8_t *p = data;

   if (!obj->pipeline_stats.results) {
      gather_statistics_results(brw, obj);

      /* Check if we did really get the results */
      if (!obj->pipeline_stats.results)
         return 0;
   }

   for (int i = 0; i < n_counters; i++) {
      *((uint64_t *)p) = obj->pipeline_stats.results[i];
      p += 8;
   }

   return p - data;
}

/**
 * Get the performance query result.
 */
static void
brw_get_perf_query_data(struct gl_context *ctx,
                        struct gl_perf_query_object *o,
                        GLsizei data_size,
                        GLuint *data,
                        GLuint *bytes_written)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   int written = 0;

   assert(brw_is_perf_query_ready(ctx, o));

   DBG("GetData(%d)\n", o->Id);

   if (INTEL_DEBUG & DEBUG_PERFMON)
      brw_dump_perf_queries(brw);

   /* This hook should only be called when results are available. */
   assert(o->Ready);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      written = get_oa_counter_data(brw, obj, data_size, (uint8_t *)data);
      break;

   case PIPELINE_STATS:
      written = get_pipeline_stats_data(brw, obj, data_size, (uint8_t *)data);
      break;
   }

   if (bytes_written)
      *bytes_written = written;
}

static struct gl_perf_query_object *
brw_new_perf_query_object(struct gl_context *ctx, int query_index)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];
   struct brw_perf_query_object *obj =
      calloc(1, sizeof(struct brw_perf_query_object));

   if (!obj)
      return NULL;

   obj->query = query;

   brw->perfquery.n_query_instances++;

   return &obj->base;
}

static void
close_perf(struct brw_context *brw)
{
   if (brw->perfquery.perf_oa_event_fd != -1) {
      if (brw->perfquery.perf_oa_mmap_base) {
         size_t mapping_len =
            brw->perfquery.perf_oa_buffer_size + brw->perfquery.page_size;

         munmap(brw->perfquery.perf_oa_mmap_base, mapping_len);
         brw->perfquery.perf_oa_mmap_base = NULL;
      }

      close(brw->perfquery.perf_oa_event_fd);
      brw->perfquery.perf_oa_event_fd = -1;
   }
}

/**
 * Delete a performance query object.
 */
static void
brw_delete_perf_query(struct gl_context *ctx,
                      struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Delete(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      if (obj->oa.bo) {
         if (!obj->oa.results_accumulated) {
            drop_from_unresolved_query_list(brw, obj);
            dec_n_oa_users(brw);
         }

         drm_intel_bo_unreference(obj->oa.bo);
         obj->oa.bo = NULL;
      }

      obj->oa.results_accumulated = false;
      break;

   case PIPELINE_STATS:
      if (obj->pipeline_stats.bo) {
         drm_intel_bo_unreference(obj->pipeline_stats.bo);
         obj->pipeline_stats.bo = NULL;
      }

      free(obj->pipeline_stats.results);
      obj->pipeline_stats.results = NULL;
      break;
   }

   free(obj);

   if (--brw->perfquery.n_query_instances == 0)
      close_perf(brw);
}

/******************************************************************************/

#define SCALED_NAMED_STAT(REG, NUM, DEN, NAME, DESC)        \
   {                                                        \
      .name = NAME,                                         \
      .desc = DESC,                                         \
      .type = GL_PERFQUERY_COUNTER_RAW_INTEL,               \
      .data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL,  \
      .size = sizeof(uint64_t),                             \
      .pipeline_stat.reg = REG,                             \
      .pipeline_stat.numerator = NUM,                       \
      .pipeline_stat.denominator = DEN,                     \
   }
#define NAMED_STAT(REG, NAME, DESC)    SCALED_NAMED_STAT(REG, 1, 1, NAME, DESC)
#define STAT(REG, DESC)                SCALED_NAMED_STAT(REG, 1, 1, #REG, DESC)
#define SCALED_STAT(REG, N, D, DESC)   SCALED_NAMED_STAT(REG, N, D, #REG, DESC)

static struct brw_perf_query_counter gen6_pipeline_statistics[] = {
   STAT(IA_VERTICES_COUNT,   "N vertices submitted"),
   STAT(IA_PRIMITIVES_COUNT, "N primitives submitted"),
   STAT(VS_INVOCATION_COUNT, "N vertex shader invocations"),
   STAT(GS_INVOCATION_COUNT, "N geometry shader invocations"),
   STAT(GS_PRIMITIVES_COUNT, "N geometry shader primitives emitted"),
   STAT(CL_INVOCATION_COUNT, "N primitives entering clipping"),
   STAT(CL_PRIMITIVES_COUNT, "N primitives leaving clipping"),
   STAT(PS_INVOCATION_COUNT, "N fragment shader invocations"),
   STAT(PS_DEPTH_COUNT,      "N z-pass fragments"),

   NAMED_STAT(GEN6_SO_PRIM_STORAGE_NEEDED, "SO_PRIM_STORAGE_NEEDED",
              "N geometry shader stream-out primitives (total)"),
   NAMED_STAT(GEN6_SO_NUM_PRIMS_WRITTEN,   "SO_NUM_PRIMS_WRITTEN",
              "N geometry shader stream-out primitives (written)"),
};

static struct brw_perf_query_counter gen7_pipeline_statistics[] = {

   STAT(IA_VERTICES_COUNT,   "N vertices submitted"),
   STAT(IA_PRIMITIVES_COUNT, "N primitives submitted"),
   STAT(VS_INVOCATION_COUNT, "N vertex shader invocations"),
   STAT(HS_INVOCATION_COUNT, "N hull shader invocations"),
   STAT(DS_INVOCATION_COUNT, "N domain shader invocations"),
   STAT(GS_INVOCATION_COUNT, "N geometry shader invocations"),
   STAT(GS_PRIMITIVES_COUNT, "N geometry shader primitives emitted"),
   STAT(CL_INVOCATION_COUNT, "N primitives entering clipping"),
   STAT(CL_PRIMITIVES_COUNT, "N primitives leaving clipping"),

   /* Implement the "WaDividePSInvocationCountBy4:HSW,BDW" workaround:
    * "Invocation counter is 4 times actual.  WA: SW to divide HW reported
    *  PS Invocations value by 4."
    *
    * Prior to Haswell, invocation count was counted by the WM, and it
    * buggily counted invocations in units of subspans (2x2 unit). To get the
    * correct value, the CS multiplied this by 4. With HSW the logic moved,
    * and correctly emitted the number of pixel shader invocations, but,
    * whomever forgot to undo the multiply by 4.
    */
   SCALED_STAT(PS_INVOCATION_COUNT, 1, 4, "N fragment shader invocations"),

   STAT(PS_DEPTH_COUNT,      "N z-pass fragments"),

   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(0), "SO_PRIM_STORAGE_NEEDED (Stream 0)",
              "N stream-out (stream 0) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(1), "SO_PRIM_STORAGE_NEEDED (Stream 1)",
              "N stream-out (stream 1) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(2), "SO_PRIM_STORAGE_NEEDED (Stream 2)",
              "N stream-out (stream 2) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(3), "SO_PRIM_STORAGE_NEEDED (Stream 3)",
              "N stream-out (stream 3) primitives (total)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(0), "SO_NUM_PRIMS_WRITTEN (Stream 0)",
              "N stream-out (stream 0) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(1), "SO_NUM_PRIMS_WRITTEN (Stream 1)",
              "N stream-out (stream 1) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(2), "SO_NUM_PRIMS_WRITTEN (Stream 2)",
              "N stream-out (stream 2) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(3), "SO_NUM_PRIMS_WRITTEN (Stream 3)",
              "N stream-out (stream 3) primitives (written)"),
};

#undef STAT
#undef NAMED_STAT

static void
add_pipeline_statistics_query(struct brw_context *brw,
                              const char *name,
                              struct brw_perf_query_counter *counters,
                              int n_counters)
{
   struct brw_perf_query *query =
      &brw->perfquery.queries[brw->perfquery.n_queries++];

   query->kind = PIPELINE_STATS;
   query->name = name;
   query->data_size = sizeof(uint64_t) * n_counters;
   query->n_counters = n_counters;
   query->counters = counters;

   for (int i = 0; i < n_counters; i++) {
      struct brw_perf_query_counter *counter = &counters[i];
      counter->offset = sizeof(uint64_t) * i;
   }
}

void
brw_init_performance_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;

   ctx->Driver.GetPerfQueryInfo = brw_get_perf_query_info;
   ctx->Driver.GetPerfCounterInfo = brw_get_perf_counter_info;
   ctx->Driver.NewPerfQueryObject = brw_new_perf_query_object;
   ctx->Driver.DeletePerfQuery = brw_delete_perf_query;
   ctx->Driver.BeginPerfQuery = brw_begin_perf_query;
   ctx->Driver.EndPerfQuery = brw_end_perf_query;
   ctx->Driver.WaitPerfQuery = brw_wait_perf_query;
   ctx->Driver.IsPerfQueryReady = brw_is_perf_query_ready;
   ctx->Driver.GetPerfQueryData = brw_get_perf_query_data;

   init_dev_info(brw);

   switch (brw->gen) {
   case 6:
      add_pipeline_statistics_query(brw, "Gen6 Pipeline Statistics Registers",
                                    gen6_pipeline_statistics,
                                    (sizeof(gen6_pipeline_statistics)/
                                     sizeof(gen6_pipeline_statistics[0])));
      break;
   case 7:
      add_pipeline_statistics_query(brw, "Gen7 Pipeline Statistics Registers",
                                    gen7_pipeline_statistics,
                                    (sizeof(gen7_pipeline_statistics)/
                                     sizeof(gen7_pipeline_statistics[0])));

      if (brw->is_haswell) {
         brw_oa_add_render_basic_counter_query_hsw(brw);
         brw_oa_add_compute_basic_counter_query_hsw(brw);
         brw_oa_add_compute_extended_counter_query_hsw(brw);
         brw_oa_add_memory_reads_counter_query_hsw(brw);
         brw_oa_add_memory_writes_counter_query_hsw(brw);
         brw_oa_add_sampler_balance_counter_query_hsw(brw);
      }
      break;
   case 8:
      add_pipeline_statistics_query(brw, "Gen8 Pipeline Statistics Registers",
                                    gen7_pipeline_statistics,
                                    (sizeof(gen7_pipeline_statistics)/
                                     sizeof(gen7_pipeline_statistics[0])));

      if (!brw->is_cherryview)
         brw_oa_add_render_basic_counter_query_bdw(brw);
      break;
   default:
      unreachable("Unexpected gen during performance queries init");
   }

   ctx->PerfQuery.NumQueries = brw->perfquery.n_queries;

   brw->perfquery.unresolved =
      ralloc_array(brw, struct brw_perf_query_object *, 2);
   brw->perfquery.unresolved_elements = 0;
   brw->perfquery.unresolved_array_size = 2;

   brw->perfquery.page_size = sysconf(_SC_PAGE_SIZE);

   brw->perfquery.perf_oa_event_fd = -1;
   brw->perfquery.perf_oa_buffer_size = 1024 * 1024; /* NB: must be power of two */

   brw->perfquery.next_query_start_report_id = 1000;
}
