################################################
#necessary imports:
################################################
import time # to monitor time execution of model
import os
import sionna

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense

# Set the number of threads to the number of CPU cores
num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

from sionna.channel import AWGN

from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims
from sionna.utils import sim_ber
from sionna.utils.plotting import PlotBER

from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr, HammingWindow
from sionna.utils import QAMSource

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver


from sionna.mapping import Mapper, Demapper, Constellation

sionna.config.seed = 42 # Set seed for reproducible random number generation


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Alternatively, try 'Agg' if you're not displaying the plot
import numpy as np
import pickle

####### Custom ###########
from ZadoffChu import ZadoffChuSequence

# Initialize timing for entire script
start_time = time.time()

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 4.0 #in sim it was 6
ebno_db_max = 18.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.75 #0.75 # Coderate for the outer code
n = 3648 #4092 #4098 #4096 Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword
num_iter = 50 #number of BP iterations 

# PTRS 
# PTRS and RPN parameters for 120 GHz
Nzc_PTRS = 4  # Length of Zadoff-Chu sequence for PTRS
Nzc_RPN = 1   # Length of Zadoff-Chu sequence for RPN
u_PTRS = 1    # Root index for PTRS
u_RPN = 2     # Root index for RPN
# both 120 and 220 
Q = 32  # Number of Q blocks


#filter
beta = 0.3 # Roll-off factor
span_in_symbols = 32 # Filter span in symbold
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor


BATCH_SIZE = 10#10 #how many examples are processed by sionna in parallel 
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)#, window ="blackman" )
rrcf.show("impulse")
rrcf.show("magnitude", "db") # Logarithmic scale
rrcf.show("magnitude", "lin") # Linear scale


qam = QAMSource(num_bits_per_symbol) # Layer to generate batches of QAM symbols


x = qam([BATCH_SIZE, num_symbols_per_codeword])

print("Shape of x", x.shape)


### PTRS and CP addition ###



### Integrate PTRS and RPN ###
# Step 1: Generate PTRS and RPN sequences
ptrs_sequence = ZadoffChuSequence(Nzc_PTRS, u_PTRS).generate_zadoff_chu()  # PTRS symbols
rpn_sequence = ZadoffChuSequence(Nzc_RPN, u_RPN).generate_zadoff_chu()  # RPN symbols
print("PTRS sequence shape is:", ptrs_sequence.shape)
print("PTRS sequence (NumPy):", ptrs_sequence)
print("rpn_sequence shape is:", rpn_sequence.shape)

# Step 2: Split QAM data into Q blocks
data_symbols_per_block = num_symbols_per_codeword // Q  # Symbols per block
data_blocks = tf.split(x, Q, axis=1)  # Split data into Q blocks (list of tensors)

print("Data symbols per block is", data_symbols_per_block)
tf.print("Tensor data blocks shape is:",tf.shape(data_blocks))

ptrs_batch = tf.tile(tf.convert_to_tensor(ptrs_sequence, dtype=tf.complex64)[tf.newaxis, :], [BATCH_SIZE, 1])
rpn_batch = tf.tile(tf.convert_to_tensor(rpn_sequence, dtype=tf.complex64)[tf.newaxis, :], [BATCH_SIZE, 1])

tf.print("PTRS shape is",tf.shape(ptrs_batch))
tf.print("RPN shape is",tf.shape(rpn_batch))
#Integrate PTRS and RPNs into Q blocks 
# Initialize an empty list to hold the blocks with PTRS and RPN
blocks_with_ptrs_rpn = []
for block in data_blocks:
    block_with_ptrs_rpn = tf.concat([ptrs_batch, rpn_batch, block], axis=1)
    blocks_with_ptrs_rpn.append(block_with_ptrs_rpn)

tf.print("Blocks with PTRS and RPNs shape is:",tf.shape(blocks_with_ptrs_rpn))

### Add cyclic prefix ###
# Define the CP ratio (e.g., 7% of block length)
cp_ratio = 0.0703125  # CP length ratio
cp_lenght = int(cp_ratio*4096 / (1-cp_ratio)) # // makes sure there are no decimal points, int needed for a tensor as well 

print("CP lenght is", cp_lenght)
# Combine all Q blocks into a single transmission block
full_block = tf.reshape(blocks_with_ptrs_rpn, [10, -1]) 
# Check the full transmission block length
tf.print("Full block shape (before CP):", tf.shape(full_block))

# Extract the CP from the last symbols of CP lenght
cp = full_block[:,-cp_lenght:]
# Prepend the CP to the full block
tf.print("CP shape is:",tf.shape(cp))

full_block_with_cp = tf.concat([cp, full_block], axis=1) #concacanetes cps and blocks, cps added at the end
# Verify the final shape
tf.print("Full block shape (with CP):", tf.shape(full_block_with_cp))

#block_with_ptrs_rpn = tf.concat([ptrs_batch, rpn_batch, first_block], axis=1)  # Shape: [10, 24]



# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)

# Upsample the QAM symbol sequence
x_us = us(x)
print("Shape of x_us", x_us.shape)

# Filter the upsampled sequence
x_rrcf = rrcf((x_us), padding = "full")
print("Shape of transmit filtered sequence x_rrcf is:",x_rrcf.shape)
print("Fiurst 10 tensor values:", x_rrcf.numpy()[:10])




# ######################################
# # ACLR constraint 
# ######################################
# #Step 1
# # Compute instantaneous power for all sequences
# instantaneous_power = np.abs(x_rrcf) ** 2  # Shape: (10, 2856)
# print("Shape of instantenous power of x_rrcf is:",instantaneous_power.shape)
# print("First 10 tensor values of power:", instantaneous_power[:10])
# #Step 2
# # Compute average power for each batch
# average_power = np.mean(instantaneous_power, axis=1)  # Shape: (10,)
# print("Shape of average power of x_rrcf is:",average_power.shape)
# print("First 10 tensor values of average power:", average_power[:10])
# #Step 3 
# #Normalize power
# # Reshape average_power to align with instantaneous_power
# normalized_power = instantaneous_power / average_power[:,None]  #  Shape: (10, 2856)
# print("Shape of anormalized power of x_rrcf is:",normalized_power.shape)
# print("First 10 tensor values of normalized power:", normalized_power[:10])
# #Step 4 
# #Convert PAPR constraint to linear scale
# papr_constraint_db = 5.5  # PAPR target in dB
# papr_constraint_linear = 10 ** (papr_constraint_db / 10)  # Linear scale
# print("Linear PARP is:", papr_constraint_linear)
# #Step 5
# #Compute PAPR violation
# violations = np.maximum(normalized_power - papr_constraint_linear, 0)  # Shape: (10, 2856)
# print("Shape of violations is:",violations.shape)
# print("First 10 tensor values of violations:", violations[:10])

# # Step 6: Aggregate violations
# average_violation = np.mean(violations, axis=1)  # Shape: (10,)
# print("Shape of average_violation is:",average_violation.shape)

# # Step 7: Clip the signal to enforce the PAPR constraint

# max_allowed_power = papr_constraint_linear * average_power  # Shape: (10, 1)
# print("Shape of max allowed power is:",max_allowed_power.shape)
# #Convert shape for maxe_allowed_power
# max_allowed_power = max_allowed_power[:, None]  # Shape becomes (10, 2856)
# print("New shape of max allowed power is:",max_allowed_power.shape)

# # Step 5.1: Create a boolean mask for symbols exceeding the max allowed power
# clipped_mask = normalized_power > papr_constraint_linear  # Shape: (10, 2856)

# # Step 5.2: Count the number of clipped symbols per batch
# clipped_count_per_batch = np.sum(clipped_mask, axis=1)  # Shape: (10,)

# # Step 5.3: Total clipped symbols across all batches
# total_clipped_symbols = np.sum(clipped_count_per_batch)

# # Print results
# for i, count in enumerate(clipped_count_per_batch):
#     print(f"Batch {i+1}: {count} symbols were clipped.")

# print(f"Total clipped symbols across all batches: {total_clipped_symbols}")

# x_rrcf_clipped = np.where(
#     normalized_power > papr_constraint_linear,
#     np.sqrt(max_allowed_power) * x_rrcf / np.abs(x_rrcf),  # Scale symbols
#     x_rrcf  # Keep symbols unchanged
# )
# print("Shape of clipped signal is:",x_rrcf_clipped.shape)

# # # Print results
# # for i, violation in enumerate(average_violation):
# #     print(f"Batch {i+1}: Average Violation = {violation:.4f}")


# Apply the matched filter
x_mf = rrcf(x_rrcf_clipped, padding = "full")
print("Shape of matched filtered sequence x_mf is:",x_mf.shape)
# Instantiate a downsampling layer
ds = Downsampling(samples_per_symbol, rrcf.length-1, num_symbols_per_codeword)
print("lenght is", rrcf.length)
# Recover the transmitted symbol sequence
x_hat = ds(x_mf)
print("shape of received signal",x_hat.shape)
# x_tensor = tf.constant(x_hat)
# #print("Tensor is ",x_tensor)
# # Apply padding
# padding_amount = tf.maximum(0, n - tf.shape(x_tensor)[2])
# paddings = tf.constant([[0,0],[0,n-int(tf.shape(x_tensor)[1])]])
# print("Padding  is",paddings)
# y_ds_padded = tf.pad(x_tensor, paddings,"CONSTANT")
# print("downsampled sequence is:",y_ds_padded.shape)


# plt.figure()
# plt.scatter(np.real(x_hat), np.imag(x_hat));
# plt.scatter(np.real(x), np.imag(x));
# plt.legend(["Transmitted", "Received"]);
# plt.title("Scatter plot of the transmitted and received QAM symbols")
# print("MSE between x and x_hat (dB)", 10*np.log10(np.var(x-x_hat)))
# plt.show()

# # Visualize the different signals
# plt.figure(figsize=(12, 8))
# plt.plot(np.real(x_us[0]), "x")
# plt.plot(np.real(x_rrcf_clipped[0, rrcf.length//2:]))
# plt.plot(np.real(x_mf[0,rrcf.length -1:]));
# plt.xlim(0,100)
# plt.legend([r"Oversampled sequence of QAM symbols $x_{us}$",
#             r"Transmitted sequence after pulse shaping $x_{rrcf}$",
#             r"Received sequence after matched filtering $x_{mf}$"]);

# plt.show()

# aclr_db = 10*np.log10(empirical_aclr(x_rrcf, oversampling=samples_per_symbol))
# print("Empirical ACLR (db):", aclr_db)
# print("Filter ACLR (dB)", 10*np.log10(rrcf.aclr))