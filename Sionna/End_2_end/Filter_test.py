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
n = 4092 #4098 #4096 Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword
num_iter = 50 #number of BP iterations 
#filter
beta = 0.25 # Roll-off factor
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

# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)

# Upsample the QAM symbol sequence
x_us = us(x)
print("Shape of x_us", x_us.shape)

# Filter the upsampled sequence
x_rrcf = rrcf((x_us), padding = "full")
print("Shape of transmit filtered sequence x_rrcf is:",x_rrcf.shape)
# Apply the matched filter
x_mf = rrcf(x_rrcf, padding = "full")
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


plt.figure()
plt.scatter(np.real(x_hat), np.imag(x_hat));
plt.scatter(np.real(x), np.imag(x));
plt.legend(["Transmitted", "Received"]);
plt.title("Scatter plot of the transmitted and received QAM symbols")
print("MSE between x and x_hat (dB)", 10*np.log10(np.var(x-x_hat)))
plt.show()

# Visualize the different signals
plt.figure(figsize=(12, 8))
plt.plot(np.real(x_us[0]), "x")
plt.plot(np.real(x_rrcf[0, rrcf.length//2:]))
plt.plot(np.real(x_mf[0,rrcf.length -1:]));
plt.xlim(0,100)
plt.legend([r"Oversampled sequence of QAM symbols $x_{us}$",
            r"Transmitted sequence after pulse shaping $x_{rrcf}$",
            r"Received sequence after matched filtering $x_{mf}$"]);

plt.show()