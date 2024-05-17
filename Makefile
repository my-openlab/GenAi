
# File: Makefile
# Author: Purush
# Date Created: February 24, 2024
# Last Modified: February 24, 2024
# Description: initial version of  cocotb Makefile
# Copyright (c) May 12, 2024 Purush
# License: All rights reserved
# Dependencies: 
# Contact: emailpurush@gmail.com
# Usage Notes: 
               

# defaults
SIM ?= verilator
TOPLEVEL_LANG ?= verilog
PWD = $(shell pwd)

# VERILOG_SOURCES = $(PWD)/rtl/multiplier_top.sv  $(PWD)/rtl/multiplier.sv 
VERILOG_SOURCES := $(wildcard $(PWD)/rtl/*.sv)
# use VHDL_SOURCES for VHDL files

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = i2s_master

# MODULE is the basename of the Python test file
MODULE = tb.test_i2s_slave

# VERILATOR_ARGS = -GCLK_DIV=64 -GWORD_SIZE=32 -GDATA_BITS=18 -GWS_EDGE=0 -cc +1800-2012ext+sv

COMPILE_ARGS += -DWS_EDGE=0  -DDATA_BITS=18
# to get trace when using verilator
EXTRA_ARGS += --trace  --trace-structs -Wno-WIDTHEXPAND
# EXTRA_ARGS += --trace-fst --trace-structs 


# verilator build directory
SIM_BUILD = build/sim_build

TRACE_DIR ?= build

TRACE_FILE ?= $(TOPLEVEL)

PLUSARGS ?= --dmpfile $(TRACE_DIR) $(TRACE_FILE)

COCOTB_RESULTS_FILE = build/results.xml

# COCOTB_HDL_TIMEUNIT = 1ns
# COCOTB_HDL_TIMEPRECISION = 1ps


# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim


.PHONY:waves

waves: i2s
	@echo
	@echo "### WAVES ###"
	gtkwave $(TRACE_DIR)/$(TOPLEVEL).vcd

i2s: cleanall 
	mkdir -p build; \
	make sim;

cleanall: clean
	rm -rf build

