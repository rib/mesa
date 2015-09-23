/*
 * Copyright (C) 2015 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * c_list_t - linked list
 *
 * The list head is of "c_list_t" type, and must be initialized
 * using c_list_init().  All entries in the list must be of the same
 * type.  The item type must have a "c_list_t" member. This
 * member will be initialized by c_list_insert(). There is no need to
 * call c_list_init() on the individual item. To query if the list is
 * empty in O(1), use c_list_empty().
 *
 * Let's call the list reference "c_list_t foo_list", the item type as
 * "item_t", and the item member as "c_list_t link". The following code
 *
 * The following code will initialize a list:
 *
 *      c_list_init(foo_list);
 *      c_list_insert(foo_list, item1);      Pushes item1 at the head
 *      c_list_insert(foo_list, item2);      Pushes item2 at the head
 *      c_list_insert(item2, item3);         Pushes item3 after item2
 *
 * The list now looks like [item2, item3, item1]
 *
 * Will iterate the list in ascending order:
 *
 *      item_t *item;
 *      c_list_for_each(item, foo_list, link) {
 *              Do_something_with_item(item);
 *      }
 */

typedef struct _c_list_t c_list_t;

struct _c_list_t {
    c_list_t *prev;
    c_list_t *next;
};

void c_list_init(c_list_t *list);
void c_list_insert(c_list_t *list, c_list_t *elm);
void c_list_remove(c_list_t *elm);
int c_list_length(c_list_t *list);
int c_list_empty(c_list_t *list);
void c_list_insert_list(c_list_t *list, c_list_t *other);

/* Only for the cool C programmers... */
#ifndef __cplusplus

/* This assigns to iterator first so that taking a reference to it
 * later in the second step won't be an undefined operation. It
 * assigns the value of list_node rather than 0 so that it is possible
 * have list_node be based on the previous value of iterator. In that
 * respect iterator is just used as a convenient temporary variable.
 * The compiler optimises all of this down to a single subtraction by
 * a constant */
#define c_list_set_iterator(list_node, iterator, member)                    \
    ((iterator) = (void *)(list_node),                                      \
     (iterator) =                                                           \
         (void *)((char *)(iterator) -                                      \
                  (((char *)&(iterator)->member) - (char *)(iterator))))

#define c_container_of(ptr, type, member)                                   \
    (type *)((char *)(ptr) - offsetof(type, member))

#define c_list_first(list, type, member)                                    \
    c_list_empty(list) ? NULL : c_container_of((list)->next, type, member);

#define c_list_last(list, type, member)                                     \
    c_list_empty(list) ? NULL : c_container_of((list)->prev, type, member);

#define c_list_for_each(pos, head, member)                                  \
    for (c_list_set_iterator((head)->next, pos, member);                    \
         &pos->member != (head);                                            \
         c_list_set_iterator(pos->member.next, pos, member))

#define c_list_for_each_safe(pos, tmp, head, member)                        \
    for (c_list_set_iterator((head)->next, pos, member),                    \
         c_list_set_iterator((pos)->member.next, tmp, member);              \
         &pos->member != (head);                                            \
         pos = tmp, c_list_set_iterator(pos->member.next, tmp, member))

#define c_list_for_each_reverse(pos, head, member)                          \
    for (c_list_set_iterator((head)->prev, pos, member);                    \
         &pos->member != (head);                                            \
         c_list_set_iterator(pos->member.prev, pos, member))

#define c_list_for_each_reverse_safe(pos, tmp, head, member)                \
    for (c_list_set_iterator((head)->prev, pos, member),                    \
         c_list_set_iterator((pos)->member.prev, tmp, member);              \
         &pos->member != (head);                                            \
         pos = tmp, c_list_set_iterator(pos->member.prev, tmp, member))

#endif /* __cplusplus */

#ifdef __cplusplus
}
#endif
