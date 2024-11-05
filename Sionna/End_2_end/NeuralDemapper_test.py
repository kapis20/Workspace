import sionna
#load required Sionna components 
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.channel import AWGN
from sionna.utils import compute_ber, compute_bler, BinarySource, ebnodb2no, hard_decisions
from sionna.mapping import Constellation, Mapper, Demapper 

#load other libraries 
import tensorflow as tf
#for DL models:
from tensorflow.keras import Model
from  tensorflow.keras.layers import Layer, Dense

import numpy as np


#needed for plots:
import matplotlib.pyplot as plt
#load custum functions 
# from Functions import calculate_ber, calculate_bler, plot_ber_bler
from Functions import plot_ber_bler



#LDPC built in Encoder/Decoder Sionna 

###Parameters from the paper:###
block_size = 4096 #N, block size - codewords lenght - must be multiple of num_bits_per_symbol
batch_size = 10 #number of codewords simulated in parallel 
code_rate = 3/4 # LDPC code rate
num_iterations = 50 #BP iterations 
inf_bits = int(block_size * code_rate) #information bits
num_bits_per_symbol = 6 #bits per modulated symbol e.q. 2^6 = 64 (QAM)
###########################################################################

#Other parameters 
#SNR range for evaluation and training (dB)
ebno_db_min = 4.0
ebno_db_max = 8.0
modulation_order = 2**num_bits_per_symbol #64 = 2^6
num_symbols_per_codeword = block_size/num_bits_per_symbol #number of modulated baseband symbols per codeword

###########################################################################
###Evaluation configuration###
results_filename = "awgn_autoencoder_results" #save results to ""


### Neural Demapper### 
class NeuralDemapper(Layer):
    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(64, 'relu') #first layer underscore indicates that attributes are private 
        self._dense_2 = Dense(64, 'relu') #second layer
        self._dense_3 = Dense(num_bits_per_symbol, None) #feature correspond to the LLRs for every bits carried by a symbol 
    def call(self, inputs):
        y,no = inputs 

        #log 10 scale to help with performance 
        no_db = log10(no)

        #stacking the real and imaginary components of the complex received samples and the noise variance
        no_db = tf.tile(no_db, [batch_size, num_symbols_per_codeword]) #[batch size, num_symbols_per_codeword]; might need to change batch size to 1 
        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y),
                      no_db], axis =2)
        
        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        llr = self._dense_3(llr)

        return llr 
    
     #Initialize the NeuralDemapper
demapper = NeuralDemapper(num_bits_per_symbol, num_symbols_per_codeword)

# Create mock inputs
y = tf.complex(tf.random.normal([batch_size, num_symbols_per_codeword]),
               tf.random.normal([batch_size, num_symbols_per_codeword]))
no = tf.random.uniform([batch_size], minval=0.1, maxval=1.0)  # Example noise values

# Call the demapper
llr = demapper([y, no])

# Print the output shape to verify
print("LLR shape:", llr.shape)  # Should be [batch_size, num_symbols_per_codeword, num_bits_per_symbol]