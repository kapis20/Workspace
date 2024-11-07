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

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
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
ebno_db_min = 6.0
ebno_db_max = 18.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.75 # Coderate for the outer code
n = 4098 #4096 Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword
num_iter = 50 #number of BP iterations 

#batch size = 10 
#Other parametes:

BATCH_SIZE = 128 #how many examples are processed by sionna in parallel 

#Name to store weights 
model_weights_path = "weights-neural-demapper"


# Dictionary to store both BER and BLER results for each model
results = {
    'ebno_dbs': {},  # SNR values for reference
    'BER': {},
    'BLER': {}
}

###############################################
#Cystom layers - Demapper 
###############################################
class NeuralDemapper(Layer): # Inherits from Keras Layer

    def __init__(self):
        super().__init__()

        # The three dense layers that form the custom trainable neural network-based demapper
        self.dense_1 = Dense(64, 'relu')
        self.dense_2 = Dense(64, 'relu')
        self.dense_3 = Dense(num_bits_per_symbol, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs

    def call(self, y):

        # y : complex-valued with shape [batch size, block length]
        # y is first mapped to a real-valued tensor with shape
        #  [batch size, block length, 2]
        # where the last dimension consists of the real and imaginary components
        # The dense layers operate on the last dimension, and treat the inner dimensions as batch dimensions, i.e.,
        # all the received symbols are independently processed.
        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z) # [batch size, number of symbols per block, number of bits per symbol]
        llr = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]
        return llr
    

    #####################################################
    #End_2_end system 
    #####################################################

class End2EndSystem(Model): # Inherits from Keras Model

    def __init__(self, training):

        super().__init__() # Must call the Keras model initializer

        self.constellation = Constellation("qam", num_bits_per_symbol, trainable=True)#, dtype = tf.complex64) # Constellation is trainable
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = NeuralDemapper() # Intantiate the NeuralDemapper custom layer as any other
        self.binary_source = BinarySource() #draw random bits to decode 
        self.awgn_channel = AWGN() #initialize awgn channel 
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function

        self.training = training

    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = ebnodb2no(ebno_db, num_bits_per_symbol,coderate) #noise variation 
        bits = self.binary_source([batch_size, n]) # Blocklength set to 1200 bits
        # Reshape bits to [batch_size, num_symbols_per_codeword, num_bits_per_symbol]
        # bits_reshaped = tf.reshape(bits, [batch_size, num_symbols_per_codeword, num_bits_per_symbol])
        x = self.mapper(bits) #symbols 
        y = self.awgn_channel([x, no]) #passed symbols to the channel together with noise variance 
        llr = self.demapper(y)  # Call the NeuralDemapper custom layer as any other
        #if training mode, then BCE loss function will be returned between binary date and llrs 
        #BCE compares how well demapper can recover the transmitted bits from received symbols 
        if self.training:
            loss = self.bce(bits, llr)
            return loss
        else:
            return bits, llr
        

###################################################
#Training - training loops as per SDG 
###################################################

# Number of iterations used for training
NUM_TRAINING_ITERATIONS = 100 #was used 30000

# Set a seed for reproducibility
tf.random.set_seed(1)

# Instantiating the end-to-end model for training
model_train = End2EndSystem(training=True)

# Adam optimizer (SGD variant)
optimizer = tf.keras.optimizers.Adam()

# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE, 6.0) #training SNR set to 6dB, get loss function for the most optimized under 6dB 
        #The model is assumed to return the BMD rate
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")

# Save the weightsin a file
weights = model_train.get_weights()
with open(model_weights_path, 'wb') as f:
    pickle.dump(weights, f)


##################################################
#model evaluation 
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
ber_NN, bler_NN = sim_ber(model, ebno_dbs, batch_size=BATCH_SIZE, num_target_block_errors=1000, max_mc_iter=1000)
results['BLER']['autoencoder-NN'] = bler_NN.numpy()
results['BER']['autoencoder-NN'] = ber_NN.numpy()

# Save the results to a file (optional)
with open("bler_results.pkl", 'wb') as f:
    pickle.dump((results), f)