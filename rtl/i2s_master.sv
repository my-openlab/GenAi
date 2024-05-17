module i2s_master #(
    parameter integer CLK_DIV = 64,      // Divide system clock to generate BCLK
    parameter integer WORD_SIZE = 32,    // Bits per word
    parameter integer DATA_BITS = 24,    // Bits per data frame
    parameter bit WS_EDGE = 1'b0         // 0: WS changes on negedge of BCLK, 1: WS changes on posedge of BCLK
)(
    input logic clk,            // System clock
    input logic rst_n,          // Active low reset
    input logic mic_data,       // Data input from the microphone
    output logic bclk,          // Bit clock output
    output logic ws,            // Word select output
    output logic [DATA_BITS-1:0] left_channel,  // Left channel audio data
    output logic [DATA_BITS-1:0] right_channel, // Right channel audio data
    output logic left_valid,    // Left channel data valid
    output logic right_valid    // Right channel data valid
);

    // Registers and wires
    logic [$clog2(CLK_DIV)-1:0] clk_div_counter;
    logic [$clog2(WORD_SIZE)-1:0] bit_counter;
    logic [DATA_BITS-1:0] left_channel_reg;
    logic [DATA_BITS-1:0] right_channel_reg;
    logic [DATA_BITS-1:0] data_buffer;
    logic left_data_ready;
    logic right_data_ready;

    // BCLK generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clk_div_counter <= 0;
            bclk <= 0;
        end else if (clk_div_counter == (CLK_DIV / 2) - 1) begin
            clk_div_counter <= 0;
            bclk <= ~bclk;
        end else begin
            clk_div_counter <= clk_div_counter + 1;
        end
    end

    // WS generation
    generate
        if (WS_EDGE) begin
            always_ff @(posedge bclk or negedge rst_n) begin
                if (!rst_n) begin
                    bit_counter <= 0;
                    ws <= 0;
                end else if (bit_counter == WORD_SIZE - 1) begin
                    bit_counter <= 0;
                    ws <= ~ws;
                end else begin
                    bit_counter <= bit_counter + 1;
                end
            end
        end else begin
            always_ff @(negedge bclk or negedge rst_n) begin
                if (!rst_n) begin
                    bit_counter <= 0;
                    ws <= 0;
                end else if (bit_counter == WORD_SIZE - 1) begin
                    bit_counter <= 0;
                    ws <= ~ws;
                end else begin
                    bit_counter <= bit_counter + 1;
                end
            end
        end
    endgenerate

    // Data capture
    always_ff @(negedge bclk or negedge rst_n) begin
        if (!rst_n) begin
            data_buffer <= 0;
            left_data_ready <= 0;
            right_data_ready <= 0;
        end else begin
            left_data_ready <= 0;
            right_data_ready <= 0;

            data_buffer <= {data_buffer[DATA_BITS-2:0], mic_data};


            if (bit_counter == DATA_BITS) begin
                if (ws) begin
                    left_channel_reg <= data_buffer;
                    left_data_ready <= 1;
                end else begin
                    right_channel_reg <= data_buffer;
                    right_data_ready <= 1;
                end
            end
        end
    end

    // Output assignments
    assign left_channel = left_channel_reg;
    assign right_channel = right_channel_reg;
    assign left_valid = left_data_ready;
    assign right_valid = right_data_ready;

endmodule
