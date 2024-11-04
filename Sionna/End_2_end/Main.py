import sionna
#load required Sionna components 
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.channel import AWGN

#load other libraries 
import tensorflow as tf
import numpy as np

#load custum functions 
from Functions import calculate_ber, calculate_bler, plot_ber_bler




#LDPC built in Encoder/Decoder Sionna 

#Parameters from the paper:
block_size = 4096 #N, block size - codewords 
code_rate = 3/4 # LDPC code rate
num_iterations = 50 #BP iterations 
inf_bits = int(block_size * code_rate) #information bits

#Other parameters 
# #generate random bits with NumPy
# input_bits = np.random.randint(0,2,inf_bits).astype(np.int32)
# #reshape input_bits to have a batch dimensions
# input_bits = input_bits.reshape(1,-1) #shape (1, inf_bits)
# #convert to TensorFlow float 32 (needed for the object in sionna)
# input_bits = tf.convert_to_tensor(input_bits, dtype = tf.float32)
snr_db = np.arange(0,11,1) #SNR values in dB arranged by 1 from 0 to 10
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

decoder = LDPC5GDecoder(
    encoder = encoder, #pass encoder instance 
    num_iter = num_iterations, #numper of BP iterations (Belief Propagation)
    hard_out = True #use hard-output decoding 
)

awgn_channel = AWGN() #init AWGN channel layer 

#Loop for differenct SNR values:
for snr_db_val in snr_db:
    #generate random bits with NumPy
    input_bits = np.random.randint(0,2,inf_bits).astype(np.int32)
    # reshape input_bits to have a batch dimensions
    input_bits = input_bits.reshape(1,-1) #shape (1, inf_bits)
    #convert to TensorFlow float 32 (needed for the object in sionna)
    input_bits = tf.convert_to_tensor(input_bits, dtype = tf.float32)

    #Encode:
    encoded_bits = encoder(input_bits)
    #convert encoded bits to cpmplex Dtype 
    encoded_bits = tf.cast(encoded_bits, dtype = tf.complex64)

    ##Convert to noise power: (needed for AWGN channel in Sionna)
    noise_power = 10** (-snr_db_val/10)

    #Transmit through AWGN channel: 
    noisy_sig = awgn_channel((encoded_bits,noise_power))

    # Convert the noisy signal to float32 (use real part)
    noisy_sig_real = tf.math.real(noisy_sig)
    noisy_sig_real = tf.cast(noisy_sig_real, dtype=tf.float32)

    #Decode received signal: 
    decoded_bits = decoder(noisy_sig_real)

    #Calculate ber and bler .numpy() to convert tensor to NumPy array
    ber = calculate_ber(input_bits,decoded_bits).numpy()
    bler = calculate_bler(input_bits,decoded_bits).numpy()

    #store results
    BER_val.append(ber)
    BLER_val.append(bler)

#Plot BLER and BER 
plot_ber_bler(snr_db,BER_val,BLER_val)


# encoded_bits = encoder(input_bits)
# #convert encoded bits to cpmplex Dtype 
# encoded_bits = tf.cast(encoded_bits, dtype = tf.complex64)
# noisy_sig = awgn_channel((encoded_bits,noise_power))

# # Convert the noisy signal to float32 (use real part)
# noisy_sig_real = tf.math.real(noisy_sig)
# noisy_sig_real = tf.cast(noisy_sig_real, dtype=tf.float32)

# decoded_bits = decoder(noisy_sig_real)

# print("Input message is:",input_bits)
# print("Decoded message is:", decoded_bits)


