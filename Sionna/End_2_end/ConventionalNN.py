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

from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr

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

#Custom functions:
from Padding_utils import PaddingFunction



###############################################
# Custom imports 
###############################################
from RappPowerAmp import RappPowerAmplifier

# Initialize timing for entire script
start_time = time.time()

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


BATCH_SIZE = tf.constant(128, tf.int32) # Training batch size#10 #how many examples are processed by sionna in parallel 
lenght_of_block = int(num_symbols_per_codeword)
#Name to store weights 
model_weights_path = "weights-neural-demapper-Conventional"
##############################################
#Evaluation metrics 
##############################################

# Dictionary to store both BER and BLER results for each model
results = {
    'ebno_dbs': {},  # SNR values for reference
    'BER': {},
    'BLER': {}
}
#Dictionary to store constellation data 
constellation_data = {
    'constellation_before': {},
    'constellation_after':{}
}

#Store loss function values, for evaluation 
loss_values = []

# # Dictionary to store the AWGN channel data before and after training
# awgn_data = {
#     "before_training": {},
#     "after_training": {}
# } Perhaps something to look into the future 

x_rrcf_signals = {}  # Dictionary to store signals for each Eb/N0 value

x_rrcf_Rapp_signals = {} #Store noisy signals 

bits_after_mapper = {} #store bits after mapper

bits_before_demapper = {}#store symbols before demapper
    
###############################################
#Cystom layers - Demapper 
###############################################
class NeuralDemapper(Layer): # Inherits from Keras Layer

    def __init__(self):
        super().__init__()

        # The three dense layers that form the custom trainable neural network-based demapper
        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        self._dense_3 = Dense(num_bits_per_symbol, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs

    def call(self, inputs):

        y,no = inputs
        # Using log10 scale helps with the performance
        no_db = log10(no)
        # Stacking the real and imaginary components of the complex received samples
        # and the noise variance
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword]) # [batch size, num_symbols_per_codeword]

        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y),
                      no_db], axis=2) # [batch size, num_symbols_per_codeword, 3]
        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        llr = self._dense_3(llr) # [batch size, num_symbols_per_codeword, num_bits_per_symbol]
        return llr
    

    #####################################################
    #End_2_end system 
    #####################################################

class End2EndSystem(Model): # Inherits from Keras Model

    def __init__(self, training):

        super().__init__() # Must call the Keras model initializer
        #####################################
        # Transmit (trainable components):
        #####################################
        self.binary_source = BinarySource() #draw random bits to decode 
        
        self.constellation = Constellation("qam", num_bits_per_symbol, trainable=True)#, dtype = tf.complex64) # Constellation is trainable
        self.mapper = Mapper(constellation=self.constellation)
        
         # Create instance of the Upsampling layer
        self.us = Upsampling(samples_per_symbol)
        #initialize the transmit filtrer 
        self.rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, trainable = True)

                    
       
        ####################################
        # Channel 
        ####################################
        self.awgn_channel = AWGN() #initialize awgn channel 
        ####################################
        # Receive
        ####################################

        # Instantiate the receive filter 
        self.m_rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
        # Instantiate a downsampling layer
        self.ds = Downsampling(samples_per_symbol, self.rrcf.length-1, num_symbols_per_codeword) #offset due to group delay
        #Demapper
        self.demapper = NeuralDemapper() # Intantiate the NeuralDemapper custom layer as any other  
        #self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function

        self.training = training

        ## conditional encoder and decoder, use only when in not training mode- other components to be added here later 
        if not self.training:
            #####################################
            # Transmit 
            ####################################
            self.interleaver = RandomInterleaver() 
           
            #initialize encoder and decoder if not in training mode 
            self.encoder = LDPC5GEncoder(k,n, num_bits_per_symbol) #pass no of info bits and lenght of the codeword, and bits per symbol
            
             ########################################
            # Non linear noise - Rapp model 
            ########################################
            self.RappModel = RappPowerAmplifier(
                saturation_amplitude = 0.9,
                smoothness_factor = 1.93
            )
            self.deinterlever = Deinterleaver(self.interleaver) #pass interlever instance
            self.decoder = LDPC5GDecoder(
                encoder=self.encoder,       # Pass encoder instance
                cn_type="boxplus-phi",      # Use stable, numerical single parity check function
                hard_out=False,             # Use soft-output decoding
                num_iter=num_iter           # Number of BP iterations
            )
        # Loss function
        if self.training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            #######################################
            # Receiver: 
            #######################################
            

          


    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        #If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        #noise variance:
        no = ebnodb2no(ebno_db, num_bits_per_symbol,coderate)
        no = expand_to_rank(no, 2)
        #########################
        ###Transmitter 
        ########################
        #Generate bits:
        #Training mode 
        if self.training:
            bits = self.binary_source([batch_size,n]) # code generates the output as if they were already encoded
            #hence n - for codeword lenght 

        else: #not training mode
            uncoded_bits = self.binary_source([batch_size,k])# smaller number of inf bits, that are later fed to encoder 
            #that produces codeword of lenght n 
            bits = self.encoder(uncoded_bits)
            bits = self.interleaver(bits)
             #############################
        
            
        # Reshape bits to [batch_size, num_symbols_per_codeword, num_bits_per_symbol]
        # bits_reshaped = tf.reshape(bits, [batch_size, num_symbols_per_codeword, num_bits_per_symbol])
        #Map bits to symbols:
        x = self.mapper(bits) #symbols 

        ############################
        #Filter and sampling
        ############################
        x_us = self.us(x) # upsampling 

        #Filter the upsampled sequence 
        x_rrcf = self.rrcf(x_us)#, padding = "same")

          ############################
        if not self.training:  # Apply Rapp model only in inference mode
           # Rapp noise addition 
            ############################
            x_rrcf = self.RappModel(x_rrcf)
            ############################
        #Channel:
        ############################
        y = self.awgn_channel([x_rrcf, no]) #passed symbols to the channel together with noise variance 

        ############################
        #matched filter, downsampling 
        ############################
        y_mf = self.m_rrcf(y)
        #y_mf = self.rrcf(y)
        y_ds = self.ds(y_mf) 
       

        ############################
        #Receiver
        ############################
        
        # Demapping 
        llr = self.demapper([y_ds,no])  # Call the NeuralDemapper custom layer as any other
        llr = tf.reshape(llr, [batch_size, n]) #Needs to be reshaped to match decoders expected inpt 
     
        ############################
        #Loss or Output
        ############################
        #if training mode, then BCE loss function will be returned between binary date and llrs 
        #BCE compares how well demapper can recover the transmitted bits from received symbols 
        if self.training:
            loss = self._bce(bits, llr)
            return loss
        else:
            #Decode llrs
            #tf.print("shape after demapping is:", tf.shape(llr))
            
            llr = self.deinterlever(llr)
            decoded_bits = self.decoder(llr)
            return uncoded_bits, decoded_bits, x_rrcf, x, y_ds
        



###################################################
#Training - training loops as per SDG 
###################################################

# Number of iterations used for training
NUM_TRAINING_ITERATIONS = 1 #was used 30000

# Set a seed for reproducibility
tf.random.set_seed(42)



# # Instantiating the end-to-end model for training
# model_train = End2EndSystem(training=True)

# # Extract and save constellation data before training
# constellation_data['constellation_before']= model_train.constellation.points.numpy()

def conventional_training(model_train):
    # Adam optimizer (SGD variant)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    # Training loop
    for i in range(NUM_TRAINING_ITERATIONS):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[BATCH_SIZE], minval=ebno_db_min, maxval=ebno_db_max)
    
        # Forward pass
        with tf.GradientTape() as tape:
            loss = model_train(BATCH_SIZE, ebno_db)
            #The model is assumed to return the BMD rate
            #store the current loss value 
            loss_values.append(loss.numpy())
        # Computing and applying gradients
        weights = model_train.trainable_weights
        grads = tape.gradient(loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
        # Print progress
        if i % 100 == 0:
            print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")


def save_weights(model_train, model_weights_path):
    weights = model_train.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)

# Instantiating the end-to-end model for training
model_train = End2EndSystem(training=True)
# Extract and save constellation data before training
constellation_data['constellation_before']= model_train.constellation.points.numpy()
# Track time for model training
training_start_time = time.time()
conventional_training(model_train)
# Save weights
save_weights(model_train, model_weights_path)
# Extract and save constellation data after training
constellation_data['constellation_after'] = model_train.constellation.points.numpy()
# End time for training
training_end_time = time.time()



# Save constellation data to a .pkl file
with open("constellation_dataNN.pkl", "wb") as f:
    pickle.dump(constellation_data, f)


# # Save the weightsin a file
# weights = model_train.get_weights()
# with open(model_weights_path, 'wb') as f:
#     pickle.dump(weights, f)

# Save loss function values to a file
with open("loss_values.pkl","wb") as f:
    pickle.dump(loss_values,f)
##################################################
#model evaluation 
##################################################

##################################################
# RUN model
##################################################

# Define the SNR range for evaluation
ebno_dbs = np.arange(ebno_db_min, ebno_db_max, 0.5)
#store the SNR values in the results array
results['ebno_dbs']['autoencoder-NN'] = ebno_dbs
# Define a function to load model weights if required
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))  # Run once to initialize
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
#model eval
model = End2EndSystem(training=False) #End2EndSystem model to run on the previously generated weights 
load_weights(model, model_weights_path)

###################################################
# SNR
###################################################

ber_NN, bler_NN = sim_ber(
    model, ebno_dbs, batch_size=BATCH_SIZE, num_target_block_errors=1, max_mc_iter=1,soft_estimates=True) #was used 1000 and 10000
    #soft estimates added for demapping 
results['BLER']['autoencoder-NN'] = bler_NN.numpy()
results['BER']['autoencoder-NN'] = ber_NN.numpy()

# Save the results to a file (optional)
with open("bler_resultsNN_conv.pkl", 'wb') as f:
    pickle.dump((results), f)


##############################################
# Signals from the model
# ########################################### 

selected_ebno_dbs = [9]  # Adjust as needed
# Evaluate model and collect signals
for ebno_db in selected_ebno_dbs:
    # Forward pass through the model
    print(f"Starting evaluation for Eb/N0 = {ebno_db} dB...")  # Print current Eb/N0
    uncoded_bits, decoded_bits, x_rrcf, x, y_ds = model(BATCH_SIZE, ebno_db)
    x_rrcf_signals[ebno_db] = x_rrcf
    bits_after_mapper[ebno_db] = x
    bits_before_demapper[ebno_db] = y_ds

print("All selected Eb/N0 evaluations completed.")



# Save the x_rrcf signals to a file (as NumPy or TF tensors)
signal_file = "x_rrcf_signals_no_clippingNN_conv.pkl"
with open(signal_file, "wb") as f:
    x_rrcf_numpy = {ebno_db: x.numpy() for ebno_db, x in x_rrcf_signals.items()}  # Convert to NumPy for storage
    pickle.dump(x_rrcf_numpy, f)


# signal_Rappfile = "x_rrcf_RappNN.pkl"
# with open(signal_Rappfile, "wb") as f:
#     x_rrcf_Rapp_numpy = {ebno_db: x.numpy() for ebno_db, x in x_rrcf_Rapp_signals.items()}  # Convert to NumPy for storage
#     pickle.dump(x_rrcf_Rapp_numpy, f)

signal_mapperFile = "x_mapperNN_conv.pkl"
with open(signal_mapperFile, "wb") as f:
    x_mapper = {ebno_db: x.numpy() for ebno_db, x in bits_after_mapper.items()}  # Convert to NumPy for storage
    pickle.dump(x_mapper, f)


signal_demapperFile = "y_demapperNN_conv.pkl"
with open(signal_demapperFile, "wb") as f:
    y_demapper = {ebno_db: x.numpy() for ebno_db, x in bits_before_demapper.items()}  # Convert to NumPy for storage
    pickle.dump(y_demapper, f)



# Time calculations: 
# Calculate and print total execution time
end_time = time.time()
total_execution_time = end_time - start_time
training_execution_time = training_end_time - training_start_time

print(f"\nTotal Execution Time: {total_execution_time:.2f} seconds")
print(f"Training Execution Time: {training_execution_time:.2f} seconds")