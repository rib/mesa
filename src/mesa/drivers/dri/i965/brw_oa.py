#!/usr/bin/env python2
#
# Copyright (c) 2015 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import xml.etree.ElementTree as ET
import argparse
import sys


def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')

c_file = None
_c_indent = 0

def c(*args):
    if c_file:
        code = ' '.join(map(str,args))
        for line in code.splitlines():
            text = ''.rjust(_c_indent) + line
            c_file.write(text.rstrip() + "\n")

def c_indent(n):
    global _c_indent
    _c_indent = _c_indent + n
def c_outdent(n):
    global _c_indent
    _c_indent = _c_indent - n

header_file = None
_h_indent = 0

def h(*args):
    if header_file:
        code = ' '.join(map(str,args))
        for line in code.splitlines():
            text = ''.rjust(_h_indent) + line
            header_file.write(text.rstrip() + "\n")

def h_indent(n):
    global _c_indent
    _h_indent = _h_indent + n
def h_outdent(n):
    global _c_indent
    _h_indent = _h_indent - n


def emit_fadd(tmp_id, args):
    c("double tmp" + str(tmp_id) +" = " + args[1] + " + " + args[0] + ";")
    return tmp_id + 1

# Be careful to check for divide by zero...
def emit_fdiv(tmp_id, args):
    c("double tmp" + str(tmp_id) +" = " + args[1] + ";")
    c("double tmp" + str(tmp_id + 1) +" = " + args[0] + ";")
    c("double tmp" + str(tmp_id + 2) +" = tmp" + str(tmp_id + 1)  + " ? tmp" + str(tmp_id) + " / tmp" + str(tmp_id + 1) + " : 0;")
    return tmp_id + 3

def emit_fmax(tmp_id, args):
    c("double tmp" + str(tmp_id) +" = " + args[1] + ";")
    c("double tmp" + str(tmp_id + 1) +" = " + args[0] + ";")
    c("double tmp" + str(tmp_id + 2) +" = MAX(tmp" + str(tmp_id) + ", tmp" + str(tmp_id + 1) + ");")
    return tmp_id + 3

def emit_fmul(tmp_id, args):
    c("double tmp" + str(tmp_id) +" = " + args[1] + " * " + args[0] + ";")
    return tmp_id + 1

def emit_fsub(tmp_id, args):
    c("double tmp" + str(tmp_id) +" = " + args[1] + " - " + args[0] + ";")
    return tmp_id + 1

def emit_read(tmp_id, args):
    type = args[1].lower()
    c("uint64_t tmp" + str(tmp_id) + " = accumulator[query->" + type + "_offset + " + args[0] + "];")
    return tmp_id + 1

def emit_uadd(tmp_id, args):
    c("uint64_t tmp" + str(tmp_id) +" = " + args[1] + " + " + args[0] + ";")
    return tmp_id + 1

# Be careful to check for divide by zero...
def emit_udiv(tmp_id, args):
    c("uint64_t tmp" + str(tmp_id) +" = " + args[1] + ";")
    c("uint64_t tmp" + str(tmp_id + 1) +" = " + args[0] + ";")
    c("uint64_t tmp" + str(tmp_id + 2) +" = tmp" + str(tmp_id + 1)  + " ? tmp" + str(tmp_id) + " / tmp" + str(tmp_id + 1) + " : 0;")
    return tmp_id + 3

def emit_umul(tmp_id, args):
    c("uint64_t tmp" + str(tmp_id) +" = " + args[1] + " * " + args[0] + ";")
    return tmp_id + 1

def emit_usub(tmp_id, args):
    c("uint64_t tmp" + str(tmp_id) +" = " + args[1] + " - " + args[0] + ";")
    return tmp_id + 1

ops = {}
# n operands, type, emitter
ops["FADD"] = (2, emit_fadd)
ops["FDIV"] = (2, emit_fdiv)
ops["FMAX"] = (2, emit_fmax)
ops["FMUL"] = (2, emit_fmul)
ops["FSUB"] = (2, emit_fsub)
ops["READ"] = (2, emit_read)
ops["UADD"] = (2, emit_uadd)
ops["UDIV"] = (2, emit_udiv)
ops["UMUL"] = (2, emit_umul)
ops["USUB"] = (2, emit_usub)

hw_vars = {}
hw_vars["$EuCoresTotalCount"] = "brw->perfquery.devinfo.n_eus"
hw_vars["$EuSlicesTotalCount"] = "brw->perfquery.devinfo.n_eu_slices"

counter_vars = {}

def output_rpn_equation_code(set, counter, equation, counter_vars):
    c("/* RPN equation: " + equation + " */")
    tokens = equation.split()
    stack = []
    tmp_id = 0
    tmp = None

    for token in tokens:
        stack.append(token)
        while stack and stack[-1] in ops:
            op = stack.pop()
            argc, callback = ops[op]
            args = []
            for i in range(0, argc):
                operand = stack.pop()
                if operand[0] == "$":
                    if operand in hw_vars:
                        operand = hw_vars[operand]
                    elif operand in counter_vars:
                        reference = counter_vars[operand]
                        operand = read_funcs[operand[1:]] + "(brw, query, accumulator)"
                    else:
                        raise Exception("Failed to resolve variable " + operand + " in equation " + equation + " for " + set.get('name') + " :: " + counter.get('name'));
                args.append(operand)

            tmp_id = callback(tmp_id, args)

            tmp = "tmp" + str(tmp_id - 1)
            stack.append(tmp)

    if tmp_id == 0:
        raise Exception("Spurious empty rpn code for " + set.get('name') + " :: " +
                counter.get('name') + ".\nThis is probably due to some unhandled RPN function, in the equation \"" +
                counter.get('equation') + "\"")

    c("\nreturn tmp" + str(tmp_id - 1) + ";")

def output_counter_read(set, counter, counter_vars):
    c("\n")
    c("/* " + set.get('name') + " :: " + counter.get('name') + " */")
    ret_type = counter.get('data_type')
    if ret_type == "uint64":
        ret_type = "uint64_t"

    c("static " + ret_type)
    read_sym = set.get('chipset').lower() + "__" + set.get('underscore_name') + "__" + counter.get('underscore_name') + "__read"
    c(read_sym + "(struct brw_context *brw,\n")
    c_indent(len(read_sym) + 1)
    c("const struct brw_perf_query *query,\n")
    c("uint64_t *accumulator)\n")
    c_outdent(len(read_sym) + 1)

    c("{")
    c_indent(3)

    output_rpn_equation_code(set, counter, counter.get('equation'), counter_vars)

    c_outdent(3)
    c("}")

    return read_sym


def output_counter_max(set, counter, counter_vars):
    max_eq = counter.get('max_equation')

    if not max_eq:
        return "0; /* undefined */"

    if max_eq == "100":
        return "100;"

    # We can only report constant maximum values via INTEL_performance_query
    for token in max_eq.split():
        if token[0] == '$' and token not in hw_vars:
            return "0; /* unsupported (varies over time) */"

    c("\n")
    c("/* " + set.get('name') + " :: " + counter.get('name') + " */")
    ret_type = counter.get('data_type')
    if ret_type == "uint64":
        ret_type = "uint64_t"

    c("static " + ret_type)
    max_sym = set.get('chipset').lower() + "__" + set.get('underscore_name') + "__" + counter.get('underscore_name') + "__max"
    c(max_sym + "(struct brw_context *brw,\n")
    c_indent(len(max_sym) + 1)
    c("struct brw_perf_query *query,\n")
    c("uint64_t *accumulator)\n")
    c_outdent(len(max_sym) + 1)

    c("{")
    c_indent(3)

    output_rpn_equation_code(set, counter, max_eq, counter_vars)

    c_outdent(3)
    c("}")

    return max_sym + "(brw);"

c_type_sizes = { "uint32_t": 4, "uint64_t": 8, "float": 4, "double": 8, "bool": 4 }
def sizeof(c_type):
    return c_type_sizes[c_type]

def pot_align(base, pot_alignment):
    return (base + pot_alignment - 1) & ~(pot_alignment - 1);

semantic_type_map = {
    "duration": "raw",
    "ratio": "event"
    }

def output_counter_report(set, counter, current_offset):
    data_type = counter.get('data_type')
    data_type_uc = data_type.upper()
    c_type = data_type

    if "uint" in c_type:
        c_type = c_type + "_t"

    semantic_type = counter.get('semantic_type')
    if semantic_type in semantic_type_map:
        semantic_type = semantic_type_map[semantic_type]

    semantic_type_uc = semantic_type.upper()

    c("\ncounter = &query->counters[query->n_counters++];\n")
    c("counter->oa_counter_read_" + data_type + " = " + read_funcs[counter.get('symbol_name')] + ";\n")
    c("counter->name = \"" + counter.get('name') + "\";\n")
    c("counter->desc = \"" + counter.get('description') + "\";\n")
    c("counter->type = GL_PERFQUERY_COUNTER_" + semantic_type_uc + "_INTEL;\n")
    c("counter->data_type = GL_PERFQUERY_COUNTER_DATA_" + data_type_uc + "_INTEL;\n")
    c("counter->raw_max = " + max_values[counter.get('symbol_name')] + "\n")

    current_offset = pot_align(current_offset, sizeof(c_type))
    c("counter->offset = " + str(current_offset) + ";\n")
    c("counter->size = sizeof(" + c_type + ");\n")

    return current_offset + sizeof(c_type)

parser = argparse.ArgumentParser()
parser.add_argument("xml", help="XML description of metrics")
parser.add_argument("--header", help="Header file to write")
parser.add_argument("--code", help="C file to write")
parser.add_argument("--include", help="Header name to include from C file")

args = parser.parse_args()

if args.header:
    header_file = open(args.header, 'w')

if args.code:
    c_file = open(args.code, 'w')

tree = ET.parse(args.xml)


copyright = """/* Autogenerated file, DO NOT EDIT manually!
 *
 * Copyright (c) 2015 Intel Corporation
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

"""

h(copyright)
h("""#pragma once

struct brw_context;

""")

c(copyright)
c(
"""
#include <stdint.h>
#include <stdbool.h>

""")

if args.include:
    c("#include \"" + args.include + "\"")

c(
"""
#include "brw_context.h"
#include "brw_performance_query.h"


#define MAX(a, b) ((a > b) ? (a) : (b))

""")

for set in tree.findall(".//set"):
    max_values = {}
    read_funcs = {}
    counter_vars = {}
    counters = set.findall("counter")
    chipset = set.get('chipset').lower()

    for counter in counters:
        empty_vars = {}
        read_funcs[counter.get('symbol_name')] = output_counter_read(set, counter, counter_vars)
        max_values[counter.get('symbol_name')] = output_counter_max(set, counter, empty_vars)
        counter_vars["$" + counter.get('symbol_name')] = counter

    h("void brw_oa_add_" + set.get('underscore_name') + "_counter_query_" + chipset + "(struct brw_context *brw);\n")

    c("\nvoid\n")
    c("brw_oa_add_" + set.get('underscore_name') + "_counter_query_" + chipset + "(struct brw_context *brw)\n")
    c("{\n")
    c_indent(3)

    c("struct brw_perf_query *query;\n")
    c("struct brw_perf_query_counter *counter;\n\n")

    c("assert(brw->perfquery.n_queries < MAX_PERF_QUERIES);\n")

    c("query = &brw->perfquery.queries[brw->perfquery.n_queries++];\n")

    c("\nquery->kind = OA_COUNTERS;\n")
    c("query->name = \"" + set.get('name') + "\";\n")

    c("query->counters = rzalloc_array(brw, struct brw_perf_query_counter, " + str(len(counters)) + ");\n")
    c("query->n_counters = 0;\n")
    c("query->oa_metrics_set = I915_OA_METRICS_SET_3D;\n")
    if chipset == "bdw":
        c("""query->oa_format = I915_OA_FORMAT_A36_B8_C8_BDW;

/* Accumulation buffer offsets... */
query->gpu_time_offset = 0;
query->gpu_clock_offset = 1;
query->a_offset = 2;
query->b_offset = query->a_offset + 36;
query->c_offset = query->b_offset + 8;

""")
    elif chipset == "hsw":
        c("""query->oa_format = I915_OA_FORMAT_A45_B8_C8_HSW;

/* Accumulation buffer offsets... */
query->gpu_time_offset = 0;
query->a_offset = 1;
query->b_offset = query->a_offset + 45;
query->c_offset = query->b_offset + 8;

""")
    else:
        assert 0

    offset = 0
    for counter in counters:
        offset = output_counter_report(set, counter, offset)


    c("\nquery->data_size = counter->offset + counter->size;\n")

    c_outdent(3)
    c("}\n")

