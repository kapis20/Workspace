import sionna
#load required Sionna components 
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.channel import AWGN
from sionna.utils import compute_ber, compute_bler, BinarySource, ebnodb2no, hard_decisions, log10, expand_to_rank, sim_ber
from sionna.mapping import Constellation, Mapper, Demapper 

#load other libraries 
import tensorflow as tf
#for DL models:
from tensorflow.keras import Model
from  tensorflow.keras.layers import Layer, Dense

import numpy as np
import pickle


#needed for plots:
import matplotlib.pyplot as plt
#load custum functions 
# from Functions import calculate_ber, calculate_bler, plot_ber_bler
from Functions import plot_ber_bler



#LDPC built in Encoder/Decoder Sionna 

###Parameters from the paper:###

block_size = 4098 # in the paper used 4096 but it is not because num_bits_per_symbol does not give it if multipleid 4096/6 = 682.6
#block_size = 4096 #N, block size - codewords lenght - must be multiple of num_bits_per_symbol
batch_size = 10 #number of codewords simulated in parallel 
code_rate = 3/4 # LDPC code rate
num_iterations = 50 #BP iterations 
inf_bits = int(block_size * code_rate) #information bits
k =inf_bits = int(block_size * code_rate) #information bits
num_bits_per_symbol = 6 #bits per modulated symbol e.q. 2^6 = 64 (QAM)
###########################################################################

#Other parameters 
#SNR range for evaluation and training (dB)
ebno_db_min = 4.0
ebno_db_max = 8.0
modulation_order = 2**num_bits_per_symbol #64 = 2^6
num_symbols_per_codeword = block_size//num_bits_per_symbol #number of modulated baseband symbols per codeword

###########################################################################
###Evaluation configuration###
results_filename = "awgn_autoencoder_results" #save results to ""




class Baseline(Model):

    def __init__(self):
        super().__init__()

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(
            k = inf_bits, #information bits (input)
            n = block_size, #coded bits numbe, codewords (output) num_bits_per_symbol)
            num_bits_per_symbol = num_bits_per_symbol
        )
        constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)

        ################
        ## Channel
        ################
        self._channel = AWGN()

        ################
        ## Receiver
        ################
        self._demapper = Demapper("app", constellation=constellation)
        self._decoder = LDPC5GDecoder(self._encoder, 
                                      num_iter = num_iterations,
                                      hard_out=True,
                                      return_infobits = True,
                                      cn_type = "boxplus-phi")



    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db, perturbation_variance=tf.constant(0.0, tf.float32)):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, code_rate) #noise variance/noise power
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        b = self._binary_source([batch_size, inf_bits])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self._channel([x, no]) # [batch size, num_symbols_per_codeword]

        ################
        ## Receiver
        ################
        llr = self._demapper([y, no])
        # Outer decoding
        b_hat = self._decoder(llr)
        return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
    

    # Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step


# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)


# Dictionary storing the results
BLER = {}

model_baseline = Baseline()
_,bler = sim_ber(model_baseline, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000)
BLER['baseline'] = bler.numpy()

# with open(results_filename, 'wb') as f:
#     pickle.dump((ebno_dbs, BLER), f)

#     plt.figure(figsize=(10,8))
# # Baseline - Perfect CSI
# plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')

# plt.xlabel(r"$E_b/N_0$ (dB)")
# plt.ylabel("BLER")
# plt.grid(which="both")
# plt.ylim((1e-4, 1.0))
# plt.legend()
# plt.tight_layout()