#This script is itended to contain various evaluation metrics and functions of the comms system 

import tensorflow as tf
import matplotlib.pyplot as plt

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


def plot_ber_bler(snr_values, ber_results, bler_results):
    # Create a figure and two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot BER on the first subplot with a logarithmic y-axis
    axs[0].plot(snr_values, ber_results, marker='o', label='BER')
    axs[0].set_xlabel('SNR (dB)')
    axs[0].set_ylabel('Bit Error Rate (BER)')
    axs[0].set_yscale('log')  # Set y-axis to logarithmic scale
    axs[0].set_title('BER vs. SNR (Log Scale)')
    axs[0].grid(True, which="both", linestyle='--', linewidth=0.5)
    axs[0].legend()

    # Plot BLER on the second subplot with a logarithmic y-axis
    axs[1].plot(snr_values, bler_results, marker='o', label='BLER', color='red')
    axs[1].set_xlabel('SNR (dB)')
    axs[1].set_ylabel('Block Error Rate (BLER)')
    axs[1].set_yscale('log')  # Set y-axis to logarithmic scale
    axs[1].set_title('BLER vs. SNR (Log Scale)')
    axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()