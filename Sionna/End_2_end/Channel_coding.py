import sionna
#load required Sionna components 
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPCDecoder
from sionna.channel import AWGN

import tensorflow as tf


#LDPC built in Encoder/Decoder Sionna 

#Parameters:
block_size = 4096 #N, block size 
code_rate = 3/4 # LDPC code rate
num_iterations = 50 #BP iterations 
inf_bits = int(block_size * code_rate) #information bits

#Initiate encoder and decoder
encoder = LDPC5GEncoder(
    k = inf_bits, #information bits
    n = block_size #coded bits number
)

decoder = LDPCDecoder(
    k = inf_bits, 
    n = block_size,
    num_iter = num_iterations, #numper of BP iterations
    hard_out = False #use soft-output decoding 
)

awgn_channel = AWGN() #init AWGN channel layer 


