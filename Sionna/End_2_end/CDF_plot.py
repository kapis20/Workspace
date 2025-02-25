import numpy as np
import matplotlib.pyplot as plt
import pickle


##################################################
# Signal files
##################################################
#NN model:
# File to save the signals
signal_file = "x_rrcf_signals_NN_conv_no_imp.pkl"
#signal_file_noisy = "x_rrcf_Rapp.pkl"
# p = 1
signal_file_noisy1="x_rrcf_signals_RAPP_p_1NN_conv.pkl"
#p = 2 
signal_file_noisy2="x_rrcf_signals_RAPP_p_2NN_conv.pkl"
#p = 3 
signal_file_noisy3="x_rrcf_signals_RAPP_p_3NN_conv.pkl"

#baseline model:
# File to save the signals
signal_file_baseline = "x_rrcf_signals_baseline_no_imp.pkl"
#signal_file_noisy = "x_rrcf_Rapp.pkl"
# p = 1
signal_file_baseline_noisy1="x_rrcf_signals_RAPP_p_1_baseline.pkl"
#p = 2 
signal_file_baseline_noisy2="x_rrcf_signals_RAPP_p_2_baseline.pkl"
#p = 3 
signal_file_baseline_noisy3="x_rrcf_signals_RAPP_p_3_baseline.pkl"

#NN model RAPP trained 
#p=1
signal_file_noisy_RAPP1="x_rrcf_signals_RAPP_trained_p_1NN_conv.pkl"

#p = 3 
signal_file_noisy_RAPP3="x_rrcf_signals_RAPP_trained_p_3NN_conv.pkl"





#########################################################
# Scaled signals BL
#########################################################
signal_file_baseline_input_scaled1="x_rrcf_signals_baseline_scaled_input_V_1.pkl"
signal_file_baseline_output_scaled1 = "x_rrcf_BL_scaled_output_V_1.pkl"

signal_file_baseline_input_scaled3="x_rrcf_signals_baseline_scaled_input_V_3.pkl"
signal_file_baseline_output_scaled3 = "x_rrcf_BL_scaled_output_V_3.pkl"

signal_file_baseline_input_scaled5="x_rrcf_signals_baseline_scaled_input_V_5.pkl"
signal_file_baseline_output_scaled5 = "x_rrcf_BL_scaled_output_V_5.pkl"

#########################################################
# Scaled NN
#########################################################
signal_file_NN_input_scaled1="x_rrcf_signals_trained_NN_conv_scaled_V_1(input).pkl"
signal_file_NN_output_scaled1 = "x_rrcf_RappNN_scaled_V_1.pkl"

signal_file_NN_input_scaled3="x_rrcf_signals_trained_NN_conv_scaled_V_3(input).pkl"
signal_file_NN_output_scaled3 = "x_rrcf_RappNN_scaled_V_3.pkl"

signal_file_NN_input_scaled5="x_rrcf_signals_trained_NN_conv_scaled_V_5(input).pkl"
signal_file_NN_output_scaled5 = "x_rrcf_RappNN_scaled_V_5.pkl"
########################################################
## Scaled BL
########################################################

with open(signal_file_baseline_input_scaled1, "rb") as f:
    Baseline_input_signal_scaled1 = pickle.load(f)

with open(signal_file_baseline_output_scaled1, "rb") as f:
    Baseline_output_signal_scaled1 = pickle.load(f)

with open(signal_file_baseline_input_scaled3, "rb") as f:
    Baseline_input_signal_scaled3 = pickle.load(f)

with open(signal_file_baseline_output_scaled3, "rb") as f:
    Baseline_output_signal_scaled3 = pickle.load(f)

with open(signal_file_baseline_input_scaled5, "rb") as f:
    Baseline_input_signal_scaled5 = pickle.load(f)

with open(signal_file_baseline_output_scaled5, "rb") as f:
    Baseline_output_signal_scaled5 = pickle.load(f)

#########################################################
## Scaled NN 
#########################################################
with open(signal_file_NN_input_scaled1, "rb") as f:
    NN_input_signal_scaled1 = pickle.load(f)

with open(signal_file_NN_output_scaled1, "rb") as f:
    NN_output_signal_scaled1 = pickle.load(f)

with open(signal_file_NN_input_scaled3, "rb") as f:
    NN_input_signal_scaled3 = pickle.load(f)

with open(signal_file_NN_output_scaled3, "rb") as f:
    NN_output_signal_scaled3 = pickle.load(f)

with open(signal_file_NN_input_scaled5, "rb") as f:
    NN_input_signal_scaled5 = pickle.load(f)

with open(signal_file_NN_output_scaled5, "rb") as f:
    NN_output_signal_scaled5 = pickle.load(f)











################################################
# Loading signals 
################################################
#NN model 

# Load signals from the file
with open(signal_file, "rb") as f:
    NNloaded_signals = pickle.load(f)


with open(signal_file_noisy1, "rb") as f:
    NNloaded_signals_noisy_p1 = pickle.load(f)

with open(signal_file_noisy2, "rb") as f:
    NNloaded_signals_noisy_p2 = pickle.load(f)


with open(signal_file_noisy3, "rb") as f:
    NNloaded_signals_noisy_p3 = pickle.load(f)


#Baseline: 

with open(signal_file_baseline, "rb") as f:
    Baseline_loaded_signals = pickle.load(f)

with open(signal_file_baseline_noisy1, "rb") as f:
    Baseline_noisy_signals_p1 = pickle.load(f)

with open(signal_file_baseline_noisy2, "rb") as f:
    Baseline_noisy_signals_p2 = pickle.load(f)

with open(signal_file_baseline_noisy3, "rb") as f:
    Baseline_noisy_signals_p3 = pickle.load(f)

##NN model RAPP trained 
with open(signal_file_noisy_RAPP1, "rb") as f:
    NNloaded_signals_noisy_RAPP_p1 = pickle.load(f)

with open(signal_file_noisy_RAPP3, "rb") as f:
    NNloaded_signals_noisy_RAPP_p3 = pickle.load(f)


def compute_cdf(signal):
    """
    Compute the CDF for a given signal.
    
    Parameters:
        signal (array-like): Input signal (can be 1D or 2D).
    
    Returns:
        x (numpy array): Sorted signal values.
        cdf (numpy array): CDF corresponding to the sorted values.
    """
    # Flatten and sort the signal, convert from 2D to 1D
    flattened_signal = signal.flatten()
    # sort in the ascending ordr 
    x = np.sort(flattened_signal)
    
    # Compute the CDF
    cdf = np.arange(1, len(x) + 1) / len(x)
    return x, cdf




signals = {
    #"E2E no impairment":NNloaded_signals[9],
    #"E2E, p=1": NNloaded_signals_noisy_p1[9],
    #"E2E, p=2": NNloaded_signals_noisy_p2[9],
    #"E2E, p=3": NNloaded_signals_noisy_p3[9],
    #"BL no impairement": Baseline_loaded_signals[9],
    # "BL, p=1": Baseline_noisy_signals_p1[9],
    # #"BL, p=2": Baseline_noisy_signals_p2[9],
    # "BL, p=3": Baseline_noisy_signals_p3[9],
    # "E2E RAPP, p=1": NNloaded_signals_noisy_RAPP_p1[9],
    # "E2E RAPP, p=3": NNloaded_signals_noisy_RAPP_p3[9]
    "BL, no impairement": Baseline_input_signal_scaled1[9][0,:],
    "BL, Vsat = 1":Baseline_output_signal_scaled1[9][0,:],
    "BL, Vsat = 3":Baseline_output_signal_scaled3[9][0,:],
    "BL, Vsat = 5":Baseline_output_signal_scaled5[9][0,:]

}


for label, signal in signals.items():
    # Compute the magnitude of the signal (complex number)
    #magnitude_signal = np.abs(signal)
    # Compute the instantaneous power (|signal|^2)
    print(f"{label}: {signal.shape}")
    instantaneous_power = np.abs(signal)**2
     # Compute the average power
    average_power = np.mean(instantaneous_power)
    # Normalize instantaneous power by average power
    normalized_power = instantaneous_power / average_power

    x, cdf = compute_cdf(normalized_power)
    plt.plot(x, cdf, label=label)


# Add labels and title

plt.xlabel("Instantaneous Power / Average Power")
plt.ylabel("CDF")
plt.title("CDFs for Multiple Signals")
plt.grid()
plt.legend()
plt.show()
# signal = Baseline_noisy_signals_p1[9]
# # Flatten the signal across all batches
# #concatenate aa the rows(batches) into a single continous array 
# flattened_signal = signal.flatten()

# # Sort the flattened signal sort in the ascending order
# x = np.sort(flattened_signal)

# # Compute CDF  
# cdf = np.arange(1, len(x) + 1) / len(x)

# # Plot the aggregated CDF
# plt.plot(x, cdf, label="Aggregated CDF")
# plt.xlabel("Signal Values")
# plt.ylabel("CDF")
# plt.title("Aggregated Empirical CDF")
# plt.grid()
# plt.legend()
# plt.show()