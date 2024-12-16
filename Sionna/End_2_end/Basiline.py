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

#sionna.config.seed = 42 # Set seed for reproducible random number generation


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Alternatively, try 'Agg' if you're not displaying the plot
import numpy as np
import pickle

###############################################
# Custom imports 
###############################################
from PN_models_new import PhaseNoise
from Cyclic_Prefix import CyclicPrefix
from PTRS_pilots import PTRSPilotInserter, PhaseNoiseCompensator
from RappPowerAmp import RappPowerAmplifier

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 6.0 #in sim it was 6
ebno_db_max = 18.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
#modulation_order = 2**num_bits_per_symbol
coderate = 0.75 #0.75 # Coderate for the outer code
n = 4092 #4098 #4096 Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword
num_iter = 50 #number of BP iterations 
#filter
beta = 0.3 # Roll-off factor
span_in_symbols = 32 # Filter span in symbold
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
#batch size = 10 
# PTRS 
# PTRS and RPN parameters for 120 GHz
Nzc_PTRS = 4  # Length of Zadoff-Chu sequence for PTRS
Nzc_RPN = 1   # Length of Zadoff-Chu sequence for RPN
u_PTRS = 1    # Root index for PTRS
u_RPN = 2     # Root index for RPN
# both 120 and 220 
Q = 32  # Number of Q blocks

# Cyclic Prefix 
cp_ratio = 0.0703125
cp_lenght = cp_ratio *n/(1+cp_ratio)

lenght_of_block = int(num_symbols_per_codeword)#+cp_lenght#Q*Nzc_PTRS)
# Phase noise 
PSD0_dB = -72, #dB
f_carrier = 120e9
f_ref = 20e9
fz = [3e4, 1.75e7]
fp = [10, 3e5]
alpha_zn = [1.4,2.55]
alpha_pn = [1.0,2.95]


#Other parametes:

BATCH_SIZE = 128#10 #how many examples are processed by sionna in parallel 

# Dictionary to store both BER and BLER results for each model
results_baseline = {
    'ebno_dbs': {},  # SNR values for reference
    'BER': {},
    'BLER': {}
}

x_rrcf_signals = {}  # Dictionary to store signals for each Eb/N0 value

x_rrcf_Rapp_signals = {} #Store noisy signals 

bits_after_mapper = {} #store bits after mapper

bits_before_demapper = {}#store symbols before demapper

#Dictionary to store constellation data 
constellation_baseline = {}
constellation_data_list = []


################################
# Temp PARP constraints function
################################
def enforce_PAPR_Constraints(x_rrcf,papr_constraint_db):
    #tf.print("Type of x_rrcf:", x_rrcf.dtype)
     # Step 1: Compute instantaneous power
    instantaneous_power = tf.abs(x_rrcf) ** 2  # Shape: (batch_size, num_symbols)
    #tf.print("Shape of instantenous power is:", tf.shape(instantaneous_power))
    # Step 2: Compute average power
    average_power = tf.reduce_mean(instantaneous_power, axis=1, keepdims=True)  # Shape: (batch_size, 1)
    #tf.print("Shape of average_power is:",tf.shape(average_power))
    # Step 3: Normalize power
    normalized_power = instantaneous_power / average_power  # Shape: (batch_size, num_symbols)
    #tf.print("Shape of normilized_power is:",tf.shape(normalized_power))
    # Step 4: Convert PAPR constraint to linear scale
    papr_constraint_linear = tf.pow(10.0, papr_constraint_db / 10.0)
    #tf.print("Shape of papr_constraint_linear is:",tf.shape(papr_constraint_linear))

    # Step 6: Count the number of clipped symbols
    clipped_mask = normalized_power > papr_constraint_linear
    clipped_count_per_batch = tf.reduce_sum(tf.cast(clipped_mask, tf.int32), axis=1)
    total_clipped_symbols = tf.reduce_sum(clipped_count_per_batch)
    #tf.print("Total clipped symbols is:",total_clipped_symbols)
    
    # Step 7: Clip the signal
    max_allowed_power = papr_constraint_linear * average_power  # Shape: (10, 1)
    #max_allowed_power = max_allowed_power[:, None]  # Shape becomes (10, 2856)
    max_allowed_power = tf.broadcast_to(max_allowed_power, tf.shape(x_rrcf))

    x_rrcf_clipped = tf.where(
    clipped_mask,
    #Explicitly Cast tf.sqrt(max_allowed_power) to complex64
    #to cast the denominator tf.abs(x_rrcf) to tf.complex64 to match the type of the numerator.
    tf.cast(tf.sqrt(max_allowed_power), tf.complex64) * x_rrcf / tf.cast(tf.abs(x_rrcf), tf.complex64),  # Clip symbols # Clip symbols
    x_rrcf  # Keep symbols unchanged
    )
    return x_rrcf_clipped

###############################
# Baseline
###############################

class Baseline(Model): # Inherits from Keras Model

    def __init__(self):

        super().__init__() # Must call the Keras model initializer
        ########################################
        # Transmitter:
        ########################################
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(k,n, num_bits_per_symbol) #pass no of info bits and lenght of the codeword, and bits per symbol
        self.interleaver = RandomInterleaver() 
        self.constellation = Constellation("qam", num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)

        ########################################
        # PTRS pilots
        ########################################
        self.PTRS = PTRSPilotInserter(
            Nzc_PTRS = Nzc_PTRS,
            u_PTRS = u_PTRS,
            Q = Q,
            batch_size = BATCH_SIZE,
            num_symbols_per_codeword = num_symbols_per_codeword
        )
        ########################################
        # Cyclic prefix 
        ########################################
        self.cp = CyclicPrefix(cp_ratio,n)

        ########################################
        # Filters
        ########################################
        # Create instance of the Upsampling layer
        self.us = Upsampling(samples_per_symbol)
        #initialize the transmit filtrer 
        self.rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
        # Instantiate a downsampling layer
        self.ds = Downsampling(samples_per_symbol, self.rrcf.length-1, lenght_of_block) #offset due to group delay

        ########################################
        # Channel 
        ########################################
        self.awgn_channel = AWGN()

        ########################################
        # Receiver 
        ########################################
        self.demapper = Demapper("app", constellation=self.constellation)
        self.deinterlever = Deinterleaver(self.interleaver) #pass interlever instance
        #self.m_rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)    
        self.decoder = LDPC5GDecoder(
                encoder=self.encoder,       # Pass encoder instance
                cn_type="boxplus-phi",      # Use stable, numerical single parity check function
                hard_out= False,             # Use soft-output decoding
                num_iter=num_iter           # Number of BP iterations
            )

        ########################################
        # Non linear noise - Rapp model 
        ########################################
        self.RappModel = RappPowerAmplifier(
            saturation_amplitude = 1,
            smoothness_factor = 3
        )

    
        ########################################
        # Phase noise 
        ########################################
        self.phase_noise = PhaseNoise(
            PSD0_dB = -72, #dB
            f_carrier = 120e9,
            f_ref = 20e9,
            fz = [3e4, 1.75e7],
            fp = [10, 3e5],
            alpha_zn = [1.4,2.55],
            alpha_pn = [1.0,2.95]
        )

        #########################################
        # Phase noise compensator 
        #########################################
        self.phase_compensator = PhaseNoiseCompensator(Nzc_PTRS)

    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = ebnodb2no(ebno_db, num_bits_per_symbol,coderate)
        uncoded_bits = self.binary_source([batch_size, k]) 
        #############################
        # Transmit
        #############################

        bits_e = self.encoder(uncoded_bits)
        bits_i = self.interleaver(bits_e)
        x = self.mapper(bits_i)
        #############################
        # PTRS pilots 
        #############################
        
        #x_PTRS, transmitted_ptrs = self.PTRS.add_ptrs_to_blocks(x)
        

        #Add cyclic prefix 
        #x_with_cp = self.cp.add_cp(x_PTRS)
        
        ############################
        #Filter and sampling
        ############################
        x_us = self.us(x) # upsampling 

        #Filter the upsampled sequence 
        x_rrcf = self.rrcf(x_us)


        #############################
        #Rapp noise addition 
        ###########################
        x_rrcf_Rapp = self.RappModel(x_rrcf)
        #tf.print("Shape of x_rrcf_Rapp is:", tf.shape(x_rrcf_Rapp))
        #tf.print("ACLR is (db)",10*np.log10(self.rrcf.aclr))
        # tf.print("Type of rrcf:", type(self.rrcf))
        # tf.print("Attributes of rrcf:", dir(self.rrcf))
        ############################
        # PAPR constraint 
        ############################
        #x_rrcf = enforce_PAPR_Constraints(x_rrcf,5.5)
        
        # Store only constellation data after mapping
        # Append ebno_db and constellation data as tuple to list
        # constellation_data_list.append((float(ebno_db), x))

        ############################
        #Phase noise - transmitter 
        ###########################
        # # Print the PSD values
        # sampling_rate = int(samples_per_symbol * 1 / (span_in_symbols / f_carrier))  # Dynamic sampling rate
        # num_samples = tf.shape(x_rrcf)[-1]
        # num_samples = tf.cast(num_samples, tf.int32)#.numpy()  # Convert to integer
        # phase_noise_samples_single = self.phase_noise.generate_phase_noise(num_samples, sampling_rate)
        # # Expand phase noise to cover all batches
    
        
        # phase_noise_samples = tf.tile(
        #     tf.expand_dims(phase_noise_samples_single, axis=0), [batch_size, 1]
        #     )  # Shape: [batch_size, num_samples]
           
      
        # # Convert phase_noise_samples to complex type
        # phase_noise_complex = tf.exp(
        #     tf.cast(1j, tf.complex64) * tf.cast(phase_noise_samples, tf.complex64)
        #     )  # Convert to complex64 phase noise
        # x_rrcf = x_rrcf * phase_noise_complex  # Apply phase noise

        ##############################
        # Channel 
        ##############################
        y = self.awgn_channel([x_rrcf_Rapp, no])
        ############################
        #matched filter, downsampling 
        ############################
        y_mf = self.rrcf(y)#, padding = "full")
        ##y_mf = self.rrcf(y)
        y_ds = self.ds(y_mf) #downsample sequence
        # ############################
        # Receive 
        ############################
        #Remove CP 
        #y_ds_no_cp = self.cp.remove_cp(y_ds)
        
        ###########################
        # PN compensation 
        ###########################
        #extract received PTRS 
        #received_blocks = tf.split(y_ds_no_cp, self.PTRS.Q, axis=1)
        
        # received_ptrs = tf.stack(
        # [block[:, :self.PTRS.Nzc_PTRS] for block in received_blocks],
        # axis=1
        #     )  # Shape: [batch_size, Q, Nzc_PTRS]
        

        # phase_error = self.phase_compensator.calculate_phase_error(received_ptrs, transmitted_ptrs)
        
        # interpolated_phase_error = self.phase_compensator.interpolate_phase_error(phase_error, tf.shape(y_ds_no_cp)[-1])
        
        # y_compensated = self.phase_compensator.compensate_phase_noise(y_ds_no_cp, interpolated_phase_error)
        

        # data_only_signal = self.phase_compensator.remove_ptrs_from_signal(y_compensated, Nzc_PTRS, Q, num_symbols_per_codeword)
        
        llr_ch = self.demapper([y_ds,no])
        #llr_rsh = tf.reshape(llr_ch, [batch_size, n]) #Needs to be reshaped to match decoders expected inpt 
        llr_de = self.deinterlever(llr_ch)
 
        llr_de = tf.reshape(llr_de, [batch_size, n]) #Needs to be reshaped to match decoders expected inpt 
        decoded_bits = self.decoder(llr_de)


        return uncoded_bits, decoded_bits, x_rrcf, x, y_ds



##################################################
#model evaluation 
##################################################



# Define the ebno range for evaluation
ebno_dbs = np.arange(ebno_db_min, ebno_db_max, 0.5)
#store the Eb/No values in the results array
results_baseline['ebno_dbs']['baseline'] = ebno_dbs

#initialize model to run 
model = Baseline()
# After evaluation, convert list to dictionary
#constellation_baseline = {ebno: data.numpy() for ebno, data in constellation_data_list}
# Specific Eb/N0 values for which signals are collected
selected_ebno_dbs = [9]  # Adjust as needed
# Evaluate model and collect signals
# for ebno_db in selected_ebno_dbs:
#     # Forward pass through the model
#     print(f"Starting evaluation for Eb/N0 = {ebno_db} dB...")  # Print current Eb/N0
#     uncoded_bits, decoded_bits, x_rrcf, x, y_ds = model(BATCH_SIZE, ebno_db)
    
#     # Save the `x_rrcf` signal (post-PAPR enforcement)
#     # Assuming `x_rrcf` is stored in the model during the forward pass
#     x_rrcf_signals[ebno_db] = x_rrcf  # Add an attribute to store `x_rrcf` in the model
#     #x_rrcf_Rapp_signals[ebno_db] = x_rrcf_Rapp
#     bits_after_mapper[ebno_db] = x
#     bits_before_demapper[ebno_db] = y_ds


# print("All selected Eb/N0 evaluations completed.")
# # Extract and save constellation data after training
# constellation_baseline['constellation_after'] = model.constellation.points.numpy()

# # Save constellation data to a .pkl file
# with open("constellation_data_QAM64.pkl", "wb") as f:
#     pickle.dump(constellation_baseline, f)




ber_NN, bler_NN = sim_ber(
    model, ebno_dbs, batch_size=BATCH_SIZE, num_target_block_errors=1000, max_mc_iter=1000,soft_estimates=True) #was used 1000 and 10000
    #soft estimates added for demapping 
results_baseline['BLER']['baseline'] = bler_NN.numpy()
results_baseline['BER']['baseline'] = ber_NN.numpy()

# Save the results to a file (optional)
with open("bler_results_baseline_P=3.pkl", 'wb') as f:
    pickle.dump(results_baseline, f)


# # Save the x_rrcf signals to a file (as NumPy or TF tensors)
# signal_file = "x_rrcf_signals_baseline_no_imp.pkl"
# with open(signal_file, "wb") as f:
#     x_rrcf_numpy = {ebno_db: x.numpy() for ebno_db, x in x_rrcf_signals.items()}  # Convert to NumPy for storage
#     pickle.dump(x_rrcf_numpy, f)


# signal_Rappfile = "x_rrcf_Rapp.pkl"
# with open(signal_Rappfile, "wb") as f:
#     x_rrcf_Rapp_numpy = {ebno_db: x.numpy() for ebno_db, x in x_rrcf_Rapp_signals.items()}  # Convert to NumPy for storage
#     pickle.dump(x_rrcf_Rapp_numpy, f)

# signal_mapperFile = "x_mapper.pkl"
# with open(signal_mapperFile, "wb") as f:
#     x_mapper = {ebno_db: x.numpy() for ebno_db, x in bits_after_mapper.items()}  # Convert to NumPy for storage
#     pickle.dump(x_mapper, f)


# signal_demapperFile = "y_demapper.pkl"
# with open(signal_demapperFile, "wb") as f:
#     y_demapper = {ebno_db: x.numpy() for ebno_db, x in bits_before_demapper.items()}  # Convert to NumPy for storage
#     pickle.dump(y_demapper, f)
