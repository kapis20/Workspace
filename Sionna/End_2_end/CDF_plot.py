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
    "E2E no impairment":NNloaded_signals[9],
    "NN, p=1": NNloaded_signals_noisy_p1[9],
    "NN, p=2": NNloaded_signals_noisy_p2[9],
    "NN, p=3": NNloaded_signals_noisy_p3[9],
    "BL no impairement": Baseline_loaded_signals[9],
    "BL, p=1": Baseline_noisy_signals_p1[9],
    "BL, p=2": Baseline_noisy_signals_p2[9],
    "BL, p=3": Baseline_noisy_signals_p3[9]

}

plt.figure(figsize=(10, 6))

for label, signal in signals.items():
    # Compute the magnitude of the signal (complex number)
    #magnitude_signal = np.abs(signal)
    # Compute the instantaneous power (|signal|^2)
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