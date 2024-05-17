import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

import cocotb.types

class I2SSlave:
    def __init__(self, dut, data_bits, word_size):
        self.dut = dut
        self.bclk = dut.bclk
        self.ws = dut.ws
        # self.sel = dut.sel
        self.data = dut.mic_data
        self.data_bits = data_bits
        self.word_size = word_size

    async def drive_data(self, left_data, right_data):
        while True:
            await RisingEdge(self.bclk)
            bit_counter = 0

            if self.ws.value.integer == 1:  # Left channel
                data = left_data
            else:  # Right channel
                data = right_data

            for bit in range(self.data_bits):
                self.data.value = (data >> (self.data_bits - 1 - bit)) & 1
                await RisingEdge(self.bclk)
                bit_counter += 1

            # Tri-state for the remaining bits in the word size
            for _ in range(self.word_size - self.data_bits-1):
                self.data.value = cocotb.types.Logic("Z")
                await RisingEdge(self.bclk)

@cocotb.test()
async def i2s_slave_test(dut):
    """ Testbench for I2S Slave interacting with I2S Master """

    DATA_BITS = int(dut.DATA_BITS)
    WORD_SIZE = int(dut.WORD_SIZE)

    # Create a 10 MHz clock on the system clock input (clk)
    cocotb.start_soon(Clock(dut.clk, 100, units='ns').start())

    # Reset
    dut.rst_n.value = 0
    # dut.sel.value = 1
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    # dut.sel.value = 1
    await ClockCycles(dut.clk, 10)

    # Predefined test data
    expected_left_data = random.randint(0, (2**DATA_BITS) - 1)
    expected_right_data = random.randint(0, (2**DATA_BITS) - 1)

    i2s_slave = I2SSlave(dut, DATA_BITS, WORD_SIZE)

    # Start driving the data
    cocotb.start_soon(i2s_slave.drive_data(expected_left_data, expected_right_data))


    # Wait for the data to be captured by the master
    await RisingEdge(dut.right_valid)
    right_data = dut.right_channel.value.integer

    # Wait for the data to be captured by the master
    await RisingEdge(dut.left_valid)
    left_data = dut.left_channel.value.integer
    await ClockCycles(dut.clk, 10)

    # Check the captured data
    print(f"expected_left_data: {hex(expected_left_data)}")
    print(f"expected_right_data: {hex(expected_right_data)}")
    assert right_data == expected_right_data, f"Right Channel Data mismatch: {hex(right_data)} != {hex(expected_right_data)}"
    assert left_data == expected_left_data, f"Left Channel Data mismatch: {hex(left_data)} != {hex(expected_left_data)}"

    print("Test Passed!")
