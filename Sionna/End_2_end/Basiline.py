################################################
#necessary imports:
################################################
import time # to monitor time execution of model
import os
import sionna

import tensorflow as tf
from tensorflow.keras import Model

# Set the number of threads to the number of CPU cores
num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

from sionna.channel import AWGN

from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims
from sionna.utils import sim_ber
from sionna.utils.plotting import PlotBER

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver

from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr

from sionna.mapping import Mapper, Demapper, Constellation

sionna.config.seed = 42 # Set seed for reproducible random number generation


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Alternatively, try 'Agg' if you're not displaying the plot
import numpy as np
import pickle


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
n = 4098 #4098 #4096 Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword
num_iter = 50 #number of BP iterations 
#filter
beta = 0.3 # Roll-off factor
span_in_symbols = 32 # Filter span in symbold
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
#batch size = 10 
#Other parametes:

BATCH_SIZE = 10#10 #how many examples are processed by sionna in parallel 

# Dictionary to store both BER and BLER results for each model
results_baseline = {
    'ebno_dbs': {},  # SNR values for reference
    'BER': {},
    'BLER': {}
}
#Dictionary to store constellation data 
constellation_baseline = {}
constellation_data_list = []

###############################
# Baseline
###############################

class Baseline(Model): # Inherits from Keras Model

    def __init__(self):

        super().__init__() # Must call the Keras model initializer

        self.constellation = Constellation("qam", num_bits_per_symbol)
        self.interleaver = RandomInterleaver() 
        self.deinterlever = Deinterleaver(self.interleaver) #pass interlever instance
        self.mapper = Mapper(constellation=self.constellation)
             # Create instance of the Upsampling layer
        self.us = Upsampling(samples_per_symbol)
        #initialize the transmit filtrer 
        self.rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
        #self.m_rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
        # Instantiate a downsampling layer
        self.ds = Downsampling(samples_per_symbol, self.rrcf.length-1, n) #offset due to group delay

        self.demapper = Demapper("app", constellation=self.constellation)
        self.binary_source = BinarySource()
        self.awgn_channel = AWGN()
        self.encoder = LDPC5GEncoder(k,n, num_bits_per_symbol) #pass no of info bits and lenght of the codeword, and bits per symbol
        self.decoder = LDPC5GDecoder(
                encoder=self.encoder,       # Pass encoder instance
                cn_type="boxplus-phi",      # Use stable, numerical single parity check function
                hard_out=False,             # Use soft-output decoding
                num_iter=num_iter           # Number of BP iterations
    )

    @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = ebnodb2no(ebno_db, num_bits_per_symbol,coderate)
        uncoded_bits = self.binary_source([batch_size, k]) 
        bits = self.encoder(uncoded_bits)
        bits = self.interleaver(bits)
        x = self.mapper(bits)
        ############################
        #Filter and sampling
        ############################
        x_us = self.us(x) # upsampling 

        #Filter the upsampled sequence 
        x_rrcf = self.rrcf(x_us)

        
        # Store only constellation data after mapping
        # Append ebno_db and constellation data as tuple to list
        # constellation_data_list.append((float(ebno_db), x))

        y = self.awgn_channel([x_rrcf, no])

        #matched filter, downsampling 
        ############################
        y_mf = self.rrcf(y, padding = "full")
        #y_mf = self.rrcf(y)
        y_ds = self.ds(y_mf) #downsample sequence

        llr = self.demapper([y_ds,no])
        #llr = tf.reshape(llr, [batch_size, n]) #Needs to be reshaped to match decoders expected inpt 
        llr = self.deinterlever(llr)
        decoded_bits = self.decoder(llr)
        return uncoded_bits, decoded_bits



##################################################
#model evaluation 
##################################################



# Define the SNR range for evaluation
ebno_dbs = np.arange(ebno_db_min, ebno_db_max, 0.5)
#store the Eb/No values in the results array
results_baseline['ebno_dbs']['baseline'] = ebno_dbs

#initialize model to run 
model = Baseline()
# After evaluation, convert list to dictionary
#constellation_baseline = {ebno: data.numpy() for ebno, data in constellation_data_list}
ber_NN, bler_NN = sim_ber(
    model, ebno_dbs, batch_size=BATCH_SIZE, num_target_block_errors=1000, max_mc_iter=1000,soft_estimates=True) #was used 1000 and 10000
    #soft estimates added for demapping 
results_baseline['BLER']['baseline'] = bler_NN.numpy()
results_baseline['BER']['baseline'] = ber_NN.numpy()

# Save the results to a file (optional)
with open("bler_results_baseline.pkl", 'wb') as f:
    pickle.dump(results_baseline, f)



# # Save constellation data to a file
# with open("constellation_baseline.pkl", 'wb') as f:
#     pickle.dump(constellation_baseline, f)