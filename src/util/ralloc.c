/*
 * Copyright © 2010 Intel Corporation
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

#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* Android defines SIZE_MAX in limits.h, instead of the standard stdint.h */
#ifdef ANDROID
#include <limits.h>
#endif

/* Some versions of MinGW are missing _vscprintf's declaration, although they
 * still provide the symbol in the import library. */
#ifdef __MINGW32__
_CRTIMP int _vscprintf(const char *format, va_list argptr);
#endif

#include "ralloc.h"

#ifndef va_copy
#ifdef __va_copy
#define va_copy(dest, src) __va_copy((dest), (src))
#else
#define va_copy(dest, src) (dest) = (src)
#endif
#endif

#define CANARY 0x5A1106

#define RALLOC_STATS_DEBUG 1


#ifdef RALLOC_STATS_DEBUG
size_t ralloc_total_watermark = 0;
size_t ralloc_user_watermark = 0;
size_t ralloc_current_total = 0;
size_t ralloc_current_user = 0;
size_t ralloc_overhead_watermark = 0;
float ralloc_efficiency_watermark = 1;
size_t ralloc_min_size = SIZE_MAX;
size_t ralloc_max_size = 0;
#endif

struct ralloc_header
{
#ifdef DEBUG
   /* A canary value used to determine whether a pointer is ralloc'd. */
   unsigned canary;
#endif

   struct ralloc_header *parent;

   /* The first child (head of a linked list) */
   struct ralloc_header *child;

   /* Linked list of siblings */
   struct ralloc_header *prev;
   struct ralloc_header *next;

   void (*destructor)(void *);

#ifdef RALLOC_STATS_DEBUG
   size_t size;
#endif
};

#define RALLOC_STATS_PRESUMED_HEADER_SIZE 40 /* ignoring debug members */

typedef struct ralloc_header ralloc_header;

static void unlink_block(ralloc_header *info);
static void unsafe_free(ralloc_header *info);

static ralloc_header *
get_header(const void *ptr)
{
   ralloc_header *info = (ralloc_header *) (((char *) ptr) -
					    sizeof(ralloc_header));
#ifdef DEBUG
   assert(info->canary == CANARY);
#endif
   return info;
}

#define PTR_FROM_HEADER(info) (((char *) info) + sizeof(ralloc_header))

static void
add_child(ralloc_header *parent, ralloc_header *info)
{
   if (parent != NULL) {
      info->parent = parent;
      info->next = parent->child;
      parent->child = info;

      if (info->next != NULL)
	 info->next->prev = info;
   }
}

void *
ralloc_context(const void *ctx)
{
   return ralloc_size(ctx, 0);
}

struct ralloc_allocator;

struct ralloc_vtable
{
   void *(*alloc)(struct ralloc_allocator *allocator,
                  const void *ctx, size_t size);
   void *(*realloc)(struct ralloc_allocator *allocator,
                    void *ptr,
                    size_t size);
   void (*free)(struct ralloc_allocator *allocator, void *ptr);
};

struct ralloc_allocator
{
   enum {
      RALLOC_TYPE_GRAPH,
      RALLOC_TYPE_STACK,
   } type;

   struct ralloc_vtable vtable;
   //memory_stack_t *stack;

   struct ralloc_allocator *prev;
};

#ifdef RALLOC_STATS_DEBUG
static void
update_watermarks(void)
{
   size_t overhead;
   float efficiency;

   if (ralloc_current_total > ralloc_total_watermark)
      ralloc_total_watermark = ralloc_current_total;
   if (ralloc_current_user > ralloc_user_watermark)
      ralloc_user_watermark = ralloc_current_user;

   overhead = ralloc_current_total - ralloc_current_user;
   if (overhead > ralloc_overhead_watermark)
      ralloc_overhead_watermark = overhead;
   efficiency = (double)ralloc_current_user / (double)ralloc_current_total;
   assert(efficiency != 0);
   if (efficiency < ralloc_efficiency_watermark)
      ralloc_efficiency_watermark = efficiency;
}

size_t sample_sizes[10001];
int n_samples = 0;

uint64_t median_accumulator;
uint64_t n_median_accumulations;

static int
sample_sort_cb(const void *v0, const void *v1)
{
   const int *i0 = v0;
   const int *i1 = v1;

   return *i0 < *i1;
}

static int *histogram;
static int histogram_len = 0;

static void sample_alloc_size(int prev_size, int size)
{
   /* The median allocation size seems more interesting than the
    * average since there are probably a small number of very
    * large allocations which we aren't interested in.
    *
    * Not wanting to log all allocation sizes up until we exit we
    * calculate the median every 1000 allocations and report the
    * average, er, median, yep.
    */
   sample_sizes[n_samples++] = size;
   if (n_samples == 1000) {
      size_t median;

      qsort(sample_sizes, 1000, sizeof(size_t), sample_sort_cb);
      n_samples = 0;

      median = sample_sizes[501];
      median_accumulator += median;
      n_median_accumulations++;
   }

   if (size > ralloc_max_size)
      ralloc_max_size = size;
   if (size < ralloc_min_size)
      ralloc_min_size = size;

#if 1
   if (!histogram || size >= histogram_len) {
      int new_len = size + 1;
      int *tail;

      histogram = realloc(histogram, new_len * sizeof(histogram[0]));

      if (histogram) {
         tail = &histogram[histogram_len];
         memset(tail, 0, (new_len - histogram_len) * sizeof(histogram[0]));
         histogram_len = new_len;
      } else {
         histogram_len = 0;
      }
   }
   if (prev_size > 0)
      histogram[prev_size]--;
   if (histogram)
      histogram[size]++;
#endif
}

__attribute__((destructor)) static void
dump_ralloc_stats(void)
{
   uint64_t size;
   int i;
   int sum = 0;

   printf("ralloc user watermark = %lu\n", ralloc_user_watermark);
   printf("ralloc total watermark = %lu\n", ralloc_total_watermark);
   printf("ralloc overhead watermark = %lu\n", ralloc_overhead_watermark);
   printf("ralloc efficiency watermark = %f\n", ralloc_efficiency_watermark);
   printf("ralloc min size = %lu\n", ralloc_min_size);
   printf("ralloc max size = %lu\n", ralloc_max_size);

   size = median_accumulator / n_median_accumulations;

   printf("average median-filtered size = %lu (check code to understand what that means)\n", size);

   printf("histogram:\n");

   for (i = 0; i < histogram_len; i++) {
      sum += histogram[i] * i;
      if (histogram[i])
         printf("%7d: = %7d allocations = %7d bytes  (sum = %d)\n", i, histogram[i], histogram[i] * i, sum);
   }
}

#endif

static void *
graph_alloc(struct ralloc_allocator *allocator,
            const void *ctx, size_t size)
{
   void *block = calloc(1, size + sizeof(ralloc_header));
   ralloc_header *info;
   ralloc_header *parent;

   if (unlikely(block == NULL))
      return NULL;

   info = (ralloc_header *) block;

   parent = ctx != NULL ? get_header(ctx) : NULL;

   add_child(parent, info);

#ifdef DEBUG
   info->canary = CANARY;
#endif
#ifdef RALLOC_STATS_DEBUG
   info->size = size;

   ralloc_current_total += RALLOC_STATS_PRESUMED_HEADER_SIZE + size;
   ralloc_current_user += size;

   sample_alloc_size(-1, size);
   update_watermarks();
#endif

   return PTR_FROM_HEADER(info);
}

static void *
graph_realloc(struct ralloc_allocator *allocator,
              void *ptr, size_t size)
{
   ralloc_header *child, *old, *info;

   old = get_header(ptr);
   info = realloc(old, size + sizeof(ralloc_header));

   if (info == NULL)
      return NULL;

#ifdef RALLOC_STATS_DEBUG
   ralloc_current_total -= 40 + info->size;
   ralloc_current_user -= info->size;
   sample_alloc_size(info->size, size);

   info->size = size;
   ralloc_current_total += 40 + size;
   ralloc_current_user += size;

   update_watermarks();
#endif

   /* Update parent and sibling's links to the reallocated node. */
   if (info != old && info->parent != NULL) {
      if (info->parent->child == old)
	 info->parent->child = info;

      if (info->prev != NULL)
	 info->prev->next = info;

      if (info->next != NULL)
	 info->next->prev = info;
   }

   /* Update child->parent links for all children */
   for (child = info->child; child != NULL; child = child->next)
      child->parent = info;

   return PTR_FROM_HEADER(info);
}

static void
graph_free(struct ralloc_allocator *allocator, void *ptr)
{
   ralloc_header *info;

   info = get_header(ptr);

#ifdef RALLOC_STATS_DEBUG
   ralloc_current_total -= 40 + info->size;
   ralloc_current_user -= info->size;

   update_watermarks();
#endif

   unlink_block(info);
   unsafe_free(info);
}

static struct ralloc_allocator graph_allocator = {
   .type = RALLOC_TYPE_GRAPH,
   .vtable.alloc = graph_alloc,
   .vtable.realloc = graph_realloc,
   .vtable.free = graph_free
};

static struct ralloc_allocator *allocator = &graph_allocator;

void *
ralloc_size(const void *ctx, size_t size)
{
   return allocator->vtable.alloc(allocator, ctx, size);
}

void *
rzalloc_size(const void *ctx, size_t size)
{
   void *ptr = ralloc_size(ctx, size);
   if (likely(ptr != NULL))
      memset(ptr, 0, size);
   return ptr;
}

/* helper function - assumes ptr != NULL */
static void *
resize(void *ptr, size_t size)
{
   return allocator->vtable.realloc(allocator, ptr, size);
}

void *
reralloc_size(const void *ctx, void *ptr, size_t size)
{
   if (unlikely(ptr == NULL))
      return ralloc_size(ctx, size);

   assert(ralloc_parent(ptr) == ctx);
   return resize(ptr, size);
}

void *
ralloc_array_size(const void *ctx, size_t size, unsigned count)
{
   if (count > SIZE_MAX/size)
      return NULL;

   return ralloc_size(ctx, size * count);
}

void *
rzalloc_array_size(const void *ctx, size_t size, unsigned count)
{
   if (count > SIZE_MAX/size)
      return NULL;

   return rzalloc_size(ctx, size * count);
}

void *
reralloc_array_size(const void *ctx, void *ptr, size_t size, unsigned count)
{
   if (count > SIZE_MAX/size)
      return NULL;

   return reralloc_size(ctx, ptr, size * count);
}

void
ralloc_free(void *ptr)
{
   if (ptr == NULL)
      return;

   allocator->vtable.free(allocator, ptr);
}

static void
unlink_block(ralloc_header *info)
{
   /* Unlink from parent & siblings */
   if (info->parent != NULL) {
      if (info->parent->child == info)
	 info->parent->child = info->next;

      if (info->prev != NULL)
	 info->prev->next = info->next;

      if (info->next != NULL)
	 info->next->prev = info->prev;
   }
   info->parent = NULL;
   info->prev = NULL;
   info->next = NULL;
}

static void
unsafe_free(ralloc_header *info)
{
   /* Recursively free any children...don't waste time unlinking them. */
   ralloc_header *temp;
   while (info->child != NULL) {
      temp = info->child;
      info->child = temp->next;
      unsafe_free(temp);
   }

   /* Free the block itself.  Call the destructor first, if any. */
   if (info->destructor != NULL)
      info->destructor(PTR_FROM_HEADER(info));

   free(info);
}

void
ralloc_steal(const void *new_ctx, void *ptr)
{
   ralloc_header *info, *parent;

   if (unlikely(ptr == NULL))
      return;

   info = get_header(ptr);
   parent = get_header(new_ctx);

   unlink_block(info);

   add_child(parent, info);
}

void
ralloc_adopt(const void *new_ctx, void *old_ctx)
{
   ralloc_header *new_info, *old_info, *child;

   if (unlikely(old_ctx == NULL))
      return;

   old_info = get_header(old_ctx);
   new_info = get_header(new_ctx);

   /* If there are no children, bail. */
   if (unlikely(old_info->child == NULL))
      return;

   /* Set all the children's parent to new_ctx; get a pointer to the last child. */
   for (child = old_info->child; child->next != NULL; child = child->next) {
      child->parent = new_info;
   }

   /* Connect the two lists together; parent them to new_ctx; make old_ctx empty. */
   child->next = new_info->child;
   new_info->child = old_info->child;
   old_info->child = NULL;
}

void *
ralloc_parent(const void *ptr)
{
   ralloc_header *info;

   if (unlikely(ptr == NULL))
      return NULL;

   info = get_header(ptr);
   return info->parent ? PTR_FROM_HEADER(info->parent) : NULL;
}

static void *autofree_context = NULL;

static void
autofree(void)
{
   ralloc_free(autofree_context);
}

void *
ralloc_autofree_context(void)
{
   if (unlikely(autofree_context == NULL)) {
      autofree_context = ralloc_context(NULL);
      atexit(autofree);
   }
   return autofree_context;
}

void
ralloc_set_destructor(const void *ptr, void(*destructor)(void *))
{
   ralloc_header *info = get_header(ptr);
   info->destructor = destructor;
}

char *
ralloc_strdup(const void *ctx, const char *str)
{
   size_t n;
   char *ptr;

   if (unlikely(str == NULL))
      return NULL;

   n = strlen(str);
   ptr = ralloc_array(ctx, char, n + 1);
   memcpy(ptr, str, n);
   ptr[n] = '\0';
   return ptr;
}

char *
ralloc_strndup(const void *ctx, const char *str, size_t max)
{
   size_t n;
   char *ptr;

   if (unlikely(str == NULL))
      return NULL;

   n = strnlen(str, max);
   ptr = ralloc_array(ctx, char, n + 1);
   memcpy(ptr, str, n);
   ptr[n] = '\0';
   return ptr;
}

/* helper routine for strcat/strncat - n is the exact amount to copy */
static bool
cat(char **dest, const char *str, size_t n)
{
   char *both;
   size_t existing_length;
   assert(dest != NULL && *dest != NULL);

   existing_length = strlen(*dest);
   both = resize(*dest, existing_length + n + 1);
   if (unlikely(both == NULL))
      return false;

   memcpy(both + existing_length, str, n);
   both[existing_length + n] = '\0';

   *dest = both;
   return true;
}


bool
ralloc_strcat(char **dest, const char *str)
{
   return cat(dest, str, strlen(str));
}

bool
ralloc_strncat(char **dest, const char *str, size_t n)
{
   /* Clamp n to the string length */
   size_t str_length = strlen(str);
   if (str_length < n)
      n = str_length;

   return cat(dest, str, n);
}

char *
ralloc_asprintf(const void *ctx, const char *fmt, ...)
{
   char *ptr;
   va_list args;
   va_start(args, fmt);
   ptr = ralloc_vasprintf(ctx, fmt, args);
   va_end(args);
   return ptr;
}

/* Return the length of the string that would be generated by a printf-style
 * format and argument list, not including the \0 byte.
 */
static size_t
printf_length(const char *fmt, va_list untouched_args)
{
   int size;
   char junk;

   /* Make a copy of the va_list so the original caller can still use it */
   va_list args;
   va_copy(args, untouched_args);

#ifdef _WIN32
   /* We need to use _vcsprintf to calculate the size as vsnprintf returns -1
    * if the number of characters to write is greater than count.
    */
   size = _vscprintf(fmt, args);
   (void)junk;
#else
   size = vsnprintf(&junk, 1, fmt, args);
#endif
   assert(size >= 0);

   va_end(args);

   return size;
}

char *
ralloc_vasprintf(const void *ctx, const char *fmt, va_list args)
{
   size_t size = printf_length(fmt, args) + 1;

   char *ptr = ralloc_size(ctx, size);
   if (ptr != NULL)
      vsnprintf(ptr, size, fmt, args);

   return ptr;
}

bool
ralloc_asprintf_append(char **str, const char *fmt, ...)
{
   bool success;
   va_list args;
   va_start(args, fmt);
   success = ralloc_vasprintf_append(str, fmt, args);
   va_end(args);
   return success;
}

bool
ralloc_vasprintf_append(char **str, const char *fmt, va_list args)
{
   size_t existing_length;
   assert(str != NULL);
   existing_length = *str ? strlen(*str) : 0;
   return ralloc_vasprintf_rewrite_tail(str, &existing_length, fmt, args);
}

bool
ralloc_asprintf_rewrite_tail(char **str, size_t *start, const char *fmt, ...)
{
   bool success;
   va_list args;
   va_start(args, fmt);
   success = ralloc_vasprintf_rewrite_tail(str, start, fmt, args);
   va_end(args);
   return success;
}

bool
ralloc_vasprintf_rewrite_tail(char **str, size_t *start, const char *fmt,
			      va_list args)
{
   size_t new_length;
   char *ptr;

   assert(str != NULL);

   if (unlikely(*str == NULL)) {
      // Assuming a NULL context is probably bad, but it's expected behavior.
      *str = ralloc_vasprintf(NULL, fmt, args);
      *start = strlen(*str);
      return true;
   }

   new_length = printf_length(fmt, args);

   ptr = resize(*str, *start + new_length + 1);
   if (unlikely(ptr == NULL))
      return false;

   vsnprintf(ptr + *start, new_length + 1, fmt, args);
   *str = ptr;
   *start += new_length;
   return true;
}
