#This script is itended to contain various evaluation metrics and functions of the comms system 

import tensorflow as tf

# Function to calculate Bit Error Rate (BER)
def calculate_ber(original_bits, decoded_bits):
    # Compare each bit in original_bits and decoded_bits
    # Cast the boolean result of the comparison to float (1 for error, 0 for no error)
    errors = tf.math.reduce_sum(tf.cast(original_bits != decoded_bits, tf.float32))
    
    # Calculate the total number of bits in original_bits
    total_bits = tf.size(original_bits, out_type=tf.float32)
    
    # Return the ratio of the number of errors to the total number of bits
    return errors / total_bits

# Function to calculate Block Error Rate (BLER)
def calculate_bler(original_bits, decoded_bits):
    # Compare each bit in original_bits and decoded_bits across each block (row)
    # Use reduce_any to check if there are any errors in each block, returning a boolean per block
    # Cast the boolean result to float (1 for block error, 0 for no block error)
    block_errors = tf.math.reduce_sum(tf.cast(tf.reduce_any(original_bits != decoded_bits, axis=1), tf.float32))
    
    # Calculate the total number of blocks (number of rows in original_bits)
    total_blocks = tf.shape(original_bits)[0]
    
    # Return the ratio of the number of block errors to the total number of blocks
    return block_errors / tf.cast(total_blocks, tf.float32)