import sionna
#load required Sionna components 
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.channel import AWGN
from sionna.utils import compute_ber, compute_bler, BinarySource, ebnodb2no, hard_decisions
from sionna.mapping import Constellation, Mapper, Demapper 

#load other libraries 
import tensorflow as tf
import numpy as np

#needed for plots:
import matplotlib.pyplot as plt
#load custum functions 
# from Functions import calculate_ber, calculate_bler, plot_ber_bler
from Functions import plot_ber_bler



#LDPC built in Encoder/Decoder Sionna 

#Parameters from the paper:
block_size = 4096 #N, block size - codewords lenght
batch_size = 10 #number of codewords simulated in parallel 
code_rate = 3/4 # LDPC code rate
num_iterations = 50 #BP iterations 
inf_bits = int(block_size * code_rate) #information bits
num_bits_per_symbol = 6 #bits per modulated symbol e.q. 2^6 = 64 (QAM)

#Other parameters 
#snr_db = np.arange(0,11,1) #SNR values in dB arranged by 1 from 0 to 10
snr_db = 5
#List to store BER and BLER resuts 
BLER_val =[]
BER_val = []


# ##Convert to noise power: (needed for AWGN channel in Sionna)
# noise_power = 10** (-snr_db/10)

###Channel encoding and decoding:###
#Initiate encoder and decoder
encoder = LDPC5GEncoder(
    k = inf_bits, #information bits (input)
    n = block_size #coded bits numbe, codewords (output)
)
#the decoder must be linked to the encoder (to know the exact code parameters for encoding)
decoder = LDPC5GDecoder(
    encoder = encoder, #pass encoder instance 
    num_iter = num_iterations, #numper of BP iterations (Belief Propagation)
    hard_out = False, #use soft-output decoding 
    return_infobits = True, # return decoded parity bits
    cn_type = "boxplus-phi" #numerical, stable single parity check
)
#initialize BinarySource class 
binary_source = BinarySource(dtype =tf.float32, seed = None) #seed none so it stays randoms

#draw random data (bits) to decode
input_bits = binary_source([batch_size,inf_bits]) #shape: (batch_size,inf_bits)

encoded_bits = encoder(input_bits)

print("Input bits are:", input_bits)
print("Shape of input bits is:",input_bits.shape)
print("Encoded bits are:",encoded_bits)
print("Shape of encoded bits is:", encoded_bits.shape)
print("Total number of processed bits is:", np.prod(encoded_bits.shape))

###Constellation and mapping:###
#initiate the constelation for 64QAM
constellation = Constellation(
    constellation_type = "qam",
    num_bits_per_symbol = num_bits_per_symbol,
    normalize = True, #true by default, but just to remember, normalization to have unit power
    trainable = True, #to allow trainable optimatizations 
    dtype =tf.complex64 #data type 

)
print("Constelattion is:")
constellation.show()
print("Constellation points are:", constellation.points.numpy())
#symbol mapping, initialize mapper: 
mapper = Mapper(constellation = constellation)
symbols = mapper(encoded_bits) #create symbols 

print("Mapped symbols are:",symbols.numpy())

### AWGN channel ###
awgn_channel = AWGN() #init AWGN channel layer 

#caluclate noise variance: 
noise_variance = ebnodb2no(snr_db,num_bits_per_symbol,code_rate)



# #Loop for differenct SNR values:
# for snr_db_val in snr_db:
#     #generate random bits with NumPy
#     input_bits = np.random.randint(0,2,inf_bits).astype(np.int32)
#     # reshape input_bits to have a batch dimensions
#     input_bits = input_bits.reshape(1,-1) #shape (1, inf_bits)
#     #convert to TensorFlow float 32 (needed for the object in sionna)
#     input_bits = tf.convert_to_tensor(input_bits, dtype = tf.float32)

#     #Encode:
#     encoded_bits = encoder(input_bits)
#     #convert encoded bits to cpmplex Dtype 
#     encoded_bits = tf.cast(encoded_bits, dtype = tf.complex64)

#     ##Convert to noise power: (needed for AWGN channel in Sionna)
#     noise_power = 10** (-snr_db_val/10)

#     #Transmit through AWGN channel: 
#     noisy_sig = awgn_channel((encoded_bits,noise_power))

#     # # Convert the noisy signal to float32 (use real part)
#     # noisy_sig_real = tf.math.real(noisy_sig)
#     # noisy_sig_real = tf.cast(noisy_sig_real, dtype=tf.float32)
#     #Convert to llrs 
#     noisy_sig_real = tf.math.real(noisy_sig) #extract real part
#     llrs = 2*noisy_sig_real / noise_power
#     #Decode received signal: 
#     decoded_bits = decoder(llrs)

#     #ensure decoded bits are binary tensor
#     decoded_bits = tf.cast(decoded_bits, tf.float32)
#     #Calculate ber and bler .numpy() to convert tensor to NumPy array
#     ber = compute_ber(input_bits,decoded_bits).numpy()
#     bler = compute_bler(input_bits,decoded_bits).numpy()

#     #store results
#     BER_val.append(ber)
#     BLER_val.append(bler)

# #Plot BLER and BER 
# plot_ber_bler(snr_db,BER_val,BLER_val)



# print("Input message is:",input_bits)
# print("Decoded message is:", decoded_bits)


