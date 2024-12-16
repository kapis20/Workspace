import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sionna.signal import RootRaisedCosineFilter, Upsampling
import pickle

# Parameters
num_bits_per_symbol = 6  # Baseline is 64-QAM
modulation_order = 2 ** num_bits_per_symbol
coderate = 0.75
n = 4092  # Codeword length [bit]
num_symbols_per_codeword = n // num_bits_per_symbol
k = int(n * coderate)
beta = 0.3
span_in_symbols = 32
samples_per_symbol = 4


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
###########################################  
# Check the loaded data
###########################################

# #NN model 
# for ebno_db, x_rrcf_signal in NNloaded_signals.items():
#     print(f"EB/N0 = {ebno_db} dB, Signal Shape: {x_rrcf_signal.shape}")
# #batch_size, num_samples = loaded_signals[9].shape  # Assuming you want EB/N0 = 8.5 dB
# signal_batch = loaded_signals[9]  # Shape: (batch_size, num_samples)
# #p = 1 
# for ebno_db, x_rrcf_signal in loaded_signals_noisy.items():
#     print(f"EB/N0 = {ebno_db} dB, Signal Shape (with Noise): {x_rrcf_signal.shape}")
# batch_size_RAPP, num_samples = loaded_signals_noisy[9].shape  # Assuming you want EB/N0 = 8.5 dB
# signal_batch_with_RAPP = loaded_signals_noisy[9]  # With RAPP
# #p = 2 

# #p = 3 



def fft(input, axis=-1):
    """
    Computes the normalized DFT along a specified axis for a NumPy array.

    This operation computes the normalized one-dimensional discrete Fourier
    transform (DFT) along the `axis` dimension of a `tensor`.
    For a vector x ∈ ℂ^N, the DFT X ∈ ℂ^N is computed as:

        X_m = (1/√N) * ∑_{n=0}^{N-1} x_n * exp(-j2πmn/N), for m=0,...,N-1.

    Parameters
    ----------
    input : np.ndarray
        Array of arbitrary shape (should be compatible with NumPy FFT).
    axis : int
        Indicates the dimension along which the DFT is taken.

    Returns
    -------
    np.ndarray
        Array of the same dtype and shape as `input` with normalized DFT applied.
    """
    # Compute the FFT size along the specified axis
    fft_size = input.shape[axis]
    
    # Compute the scale factor
    scale = 1 / np.sqrt(fft_size)
    
    # Compute the FFT along the specified axis
    output = np.fft.fft(input, axis=axis)
    
    # Apply the normalization scale
    return scale * output




def empirical_psd(x, show=True, oversampling=1.0, ylim=(-30, 3)):
    r"""
    Computes the empirical power spectral density (PSD) of a NumPy array.

    Computes the empirical power spectral density (PSD) of array ``x``
    along the last dimension by averaging over all other dimensions.
    This function returns the averaged absolute squared discrete Fourier
    spectrum of ``x``.

    Parameters
    ----------
    x : np.ndarray, [..., N], complex
        The signal for which to compute the PSD.

    show : bool
        Indicates if a plot of the PSD should be generated.
        Defaults to True.

    oversampling : float
        The oversampling factor. Defaults to 1.

    ylim : tuple of floats
        The limits of the y-axis. Defaults to [-30, 3].
        Only relevant if ``show`` is True.

    Returns
    -------
    freqs : np.ndarray, [N], float
        The normalized frequencies at which the PSD was evaluated.

    psd : np.ndarray, [N], float
        The PSD.
    """
    # Compute the FFT and the PSD
    fft_result = fft(x)
    psd = np.abs(fft_result) ** 2
    
    # Average over all dimensions except the last
    psd = np.mean(psd, axis=tuple(range(x.ndim - 1)))
    
    # Apply FFT shift for proper frequency ordering
    psd = np.fft.fftshift(psd)
    
    # Create normalized frequency vector - numper of the last dimesnons
    N = x.shape[-1]
    f_min = -0.5 * oversampling
    f_max = -f_min
    freqs = np.linspace(f_min, f_max, N)
    
    # Plot the PSD if required
    if show:
        plt.figure()
        plt.plot(freqs, 10 * np.log10(psd))
        plt.title("Power Spectral Density")
        plt.xlabel("Normalized Frequency")
        plt.xlim([freqs[0], freqs[-1]])
        plt.ylabel(r"$\mathbb{E}\left[|X(f)|^2\right]$ (dB)")
        plt.ylim(ylim)
        plt.grid(True, which="both")
        plt.show()

    return freqs, psd

def find_3db_points(freqs, psd):
    """
    Find the -3 dB points in the PSD.
    """
    psd_max = np.max(psd)
    threshold = psd_max / 2  # -3 dB corresponds to half of max power
    indices = np.where(psd >= threshold)[0]  # Indices where PSD is above threshold
    f_low, f_high = freqs[indices[0]], freqs[indices[-1]]
    #print("f_low is", f_low)
    #print("f high is", f_high)
    return f_low, f_high

def firstSlopeBW(freqs,psd, threshold_dB):
    """
    Find emperical for BW based on the first slope of the signal
    """
    #Convert threshold to dB
    threshold = 10**(threshold_dB/10)
    indices = np.where(psd >= threshold)[0]  # Indices where PSD is above threshold
    f_low, f_high = freqs[indices[0]], freqs[indices[-1]]
    #print("f_low is", f_low)
    #print("f high is", f_high)
    return f_low, f_high

def compute_aclr_3dB(freqs, psd):
    """
    Compute the ACLR as the ratio of main channel power to adjacent channel power.
    """

    f_low, f_high = find_3db_points(freqs,psd)
    # Power in the main channel
    main_channel_mask = (freqs >= f_low) & (freqs <= f_high)
    P_main = np.sum(psd[main_channel_mask])
    
    # Define adjacent channels with the same bandwidth as the main channel
    bandwidth = f_high - f_low
    #print("bandwith is: ", bandwidth)

    left_adjacent_mask = (freqs >= f_low - bandwidth) & (freqs < f_low)
    #print("left adcj mask is", f_low - bandwidth)
    right_adjacent_mask = (freqs > f_high) & (freqs <= f_high + bandwidth)
    
    # Power in adjacent channels
    P_adjacent = np.sum(psd[left_adjacent_mask]) + np.sum(psd[right_adjacent_mask])
    
    # ACLR computation
    aclr = P_adjacent/P_main
    aclr_db = 10*np.log10(aclr)

    return aclr_db


def compute_aclr_first_slope(freqs, psd, threshold_dB):
    """
    Compute the ACLR as the ratio of main channel power to adjacent channel power.
    """

    f_low, f_high = firstSlopeBW(freqs,psd, threshold_dB)
    # Power in the main channel
    main_channel_mask = (freqs >= f_low) & (freqs <= f_high)
    P_main = np.sum(psd[main_channel_mask])
    
    # Define adjacent channels with the same bandwidth as the main channel
    bandwidth = f_high - f_low
    #print("bandwith 1st sloper is: ", bandwidth)

    left_adjacent_mask = (freqs >= f_low - bandwidth) & (freqs < f_low)
    #print("left adcj mask is", f_low - bandwidth)
    right_adjacent_mask = (freqs > f_high) & (freqs <= f_high + bandwidth)
    
    # Power in adjacent channels
    P_adjacent = np.sum(psd[left_adjacent_mask]) + np.sum(psd[right_adjacent_mask])
    
    # ACLR computation
    aclr = P_adjacent/P_main
    aclr_db = 10*np.log10(aclr)

    return aclr_db

def SionnaACLR(freqs,psd,f_low=-0.5, f_high=0.5):
    main_channel_mask = (freqs >= f_low) & (freqs <= f_high)
    P_main = np.sum(psd[main_channel_mask])
    # Define adjacent channels with the same bandwidth as the main channel
    bandwidth = f_high - f_low
    #print("bandwith Sionna is: ", bandwidth)
    left_adjacent_mask = (freqs >= f_low - bandwidth) & (freqs < f_low)
    #print("left adcj mask is", f_low - bandwidth)
    right_adjacent_mask = (freqs > f_high) & (freqs <= f_high + bandwidth)
    # Power in adjacent channels
    P_adjacent = np.sum(psd[left_adjacent_mask]) + np.sum(psd[right_adjacent_mask])
    # ACLR computation
    aclr =  P_adjacent/P_main
    aclr_db = 10*np.log10(aclr)
    return aclr_db

######################################################
# Store values
######################################################
ACLR_3dB_BL =[]
ACLR_3dB_NN =[]
ACLR_Sionna_BL =[]
ACLR_Sionna_NN = []
ACLR_1st_slope_BL = []
ACLR_1st_slope_NN = []

# Smoothness factors (1 to 3 for [1-3], with [0] for no RAPP PA)
smoothness_factors = [0, 1, 2, 3]


ylim=(-110,3)
#NN model 
freqs_NN_no_imp, psd_NN_no_imp = empirical_psd(NNloaded_signals[9], show = False, oversampling = 4.0, ylim=ylim)

freqs_NN_p_1, psd_NN_p_1 = empirical_psd(NNloaded_signals_noisy_p1[9], show = False, oversampling = 4.0, ylim=ylim)

freqs_NN_p_2, psd_NN_p_2 = empirical_psd(NNloaded_signals_noisy_p2[9], show = False, oversampling = 4.0, ylim=ylim)

freqs_NN_p_3, psd_NN_p_3 = empirical_psd(NNloaded_signals_noisy_p3[9], show = False, oversampling = 4.0, ylim=ylim)

#Baseline 
freqs_Baseline_no_imp, psd_Baseline_no_imp = empirical_psd(Baseline_loaded_signals[9], show = False, oversampling = 4.0, ylim=ylim)

freqs_Baseline_p_1, psd_Baseline_p_1 = empirical_psd(Baseline_noisy_signals_p1[9], show = False, oversampling = 4.0, ylim=ylim)

freqs_Baseline_p_2, psd_Baseline_p_2 = empirical_psd(Baseline_noisy_signals_p2[9], show = False, oversampling = 4.0, ylim=ylim)

freqs_Baseline_p_3, psd_Baseline_p_3 = empirical_psd(Baseline_noisy_signals_p3[9], show = False, oversampling = 4.0, ylim=ylim)

# #find_3db_points(freqs_NN_no_imp,psd_NN_no_imp)
# print("ACLR 3dB is",compute_aclr_3dB(freqs_NN_no_imp,psd_NN_no_imp))
# print("ACLR first slope is",compute_aclr_first_slope(freqs_NN_no_imp,psd_NN_no_imp,-45))
# print("ACLR Sionna is",SionnaACLR(freqs_NN_no_imp,psd_NN_no_imp))

###################################################
# Calculate ACLR 
###################################################

import pandas as pd

# Replace these with your computed values
ACLR_3dB_BL = [
    compute_aclr_3dB(freqs_Baseline_no_imp, psd_NN_no_imp),
    compute_aclr_3dB(freqs_Baseline_p_1, psd_Baseline_p_1),
    compute_aclr_3dB(freqs_Baseline_p_2, psd_Baseline_p_2),
    compute_aclr_3dB(freqs_Baseline_p_3, psd_Baseline_p_3),
]
ACLR_3dB_NN = [
    compute_aclr_3dB(freqs_NN_no_imp, psd_Baseline_no_imp),
    compute_aclr_3dB(freqs_NN_p_1, psd_NN_p_1),
    compute_aclr_3dB(freqs_NN_p_2, psd_NN_p_2),
    compute_aclr_3dB(freqs_NN_p_3, psd_NN_p_3),
]
ACLR_Sionna_BL = [
    SionnaACLR(freqs_Baseline_no_imp, psd_Baseline_no_imp),
    SionnaACLR(freqs_Baseline_p_1, psd_Baseline_p_1),
    SionnaACLR(freqs_Baseline_p_2, psd_Baseline_p_2),
    SionnaACLR(freqs_Baseline_p_3, psd_Baseline_p_3),
]
ACLR_Sionna_NN = [
    SionnaACLR(freqs_NN_no_imp, psd_NN_no_imp),
    SionnaACLR(freqs_NN_p_1, psd_NN_p_1),
    SionnaACLR(freqs_NN_p_2, psd_NN_p_2),
    SionnaACLR(freqs_NN_p_3, psd_NN_p_3),
]
ACLR_1st_slope_BL = [
    compute_aclr_first_slope(freqs_Baseline_no_imp, psd_Baseline_no_imp, -45),
    compute_aclr_first_slope(freqs_Baseline_p_1, psd_Baseline_p_1, -45),
    compute_aclr_first_slope(freqs_Baseline_p_2, psd_Baseline_p_2, -45),
    compute_aclr_first_slope(freqs_Baseline_p_3, psd_Baseline_p_3, -45),
]
ACLR_1st_slope_NN = [
    compute_aclr_first_slope(freqs_NN_no_imp, psd_NN_no_imp, -45),
    compute_aclr_first_slope(freqs_NN_p_1, psd_NN_p_1, -45),
    compute_aclr_first_slope(freqs_NN_p_2, psd_NN_p_2, -45),
    compute_aclr_first_slope(freqs_NN_p_3, psd_NN_p_3, -45),
]

# Combine values into a table
data = {
    "System": [
        "BL (no RAPP)", "BL (p=1)", "BL (p=2)", "BL (p=3)",
        "E2E (no RAPP)", "E2E (p=1)", "E2E (p=2)", "E2E (p=3)"
    ],
    "3dB Method": ACLR_3dB_BL + ACLR_3dB_NN,
    "Sionna Method": ACLR_Sionna_BL + ACLR_Sionna_NN,
    "1st Slope Method": ACLR_1st_slope_BL + ACLR_1st_slope_NN,
}

# Create a DataFrame
table_df = pd.DataFrame(data)

# Display the table
print(table_df)

# Save the table to a CSV file (optional)
table_df.to_csv("ACLR_Comparison_Table.csv", index=False)
# Save the table as a LaTeX file
table_df.to_latex("ACLR_Comparison_Table.tex", index=False, caption="ACLR Comparison Table", label="tab:aclr_table")













#1 3dB 
#BL 
# # Dynamically append values
# ACLR_3dB_BL.append(compute_aclr_3dB(freqs_Baseline_no_imp, psd_NN_no_imp))
# ACLR_3dB_BL.append(compute_aclr_3dB(freqs_Baseline_p_1, psd_Baseline_p_1))
# ACLR_3dB_BL.append(compute_aclr_3dB(freqs_Baseline_p_2, psd_Baseline_p_2))
# ACLR_3dB_BL.append(compute_aclr_3dB(freqs_Baseline_p_3, psd_Baseline_p_3))
# #NN
# ACLR_3dB_NN.append(compute_aclr_3dB(freqs_NN_no_imp, psd_Baseline_no_imp))
# ACLR_3dB_NN.append(compute_aclr_3dB(freqs_NN_p_1, psd_NN_p_1))
# ACLR_3dB_NN.append(compute_aclr_3dB(freqs_NN_p_2, psd_NN_p_2))
# ACLR_3dB_NN.append(compute_aclr_3dB(freqs_NN_p_3, psd_NN_p_3))

# #2
# # Compute ACLR_Sionna_BL
# ACLR_Sionna_BL.append(SionnaACLR(freqs_Baseline_no_imp, psd_Baseline_no_imp))
# ACLR_Sionna_BL.append(SionnaACLR(freqs_Baseline_p_1, psd_Baseline_p_1))
# ACLR_Sionna_BL.append(SionnaACLR(freqs_Baseline_p_2, psd_Baseline_p_2))
# ACLR_Sionna_BL.append(SionnaACLR(freqs_Baseline_p_3, psd_Baseline_p_3))

# # Compute ACLR_Sionna_NN
# ACLR_Sionna_NN.append(SionnaACLR(freqs_NN_no_imp, psd_NN_no_imp))
# ACLR_Sionna_NN.append(SionnaACLR(freqs_NN_p_1, psd_NN_p_1))
# ACLR_Sionna_NN.append(SionnaACLR(freqs_NN_p_2, psd_NN_p_2))
# ACLR_Sionna_NN.append(SionnaACLR(freqs_NN_p_3, psd_NN_p_3))

# #3
# # Compute ACLR_1st_slope_BL
# ACLR_1st_slope_BL.append(compute_aclr_first_slope(freqs_Baseline_no_imp, psd_Baseline_no_imp,-45))
# ACLR_1st_slope_BL.append(compute_aclr_first_slope(freqs_Baseline_p_1, psd_Baseline_p_1,-45))
# ACLR_1st_slope_BL.append(compute_aclr_first_slope(freqs_Baseline_p_2, psd_Baseline_p_2,-45))
# ACLR_1st_slope_BL.append(compute_aclr_first_slope(freqs_Baseline_p_3, psd_Baseline_p_3,-45))

# # Compute ACLR_1st_slope_NN
# ACLR_1st_slope_NN.append(compute_aclr_first_slope(freqs_NN_no_imp, psd_NN_no_imp,-45))
# ACLR_1st_slope_NN.append(compute_aclr_first_slope(freqs_NN_p_1, psd_NN_p_1,-45))
# ACLR_1st_slope_NN.append(compute_aclr_first_slope(freqs_NN_p_2, psd_NN_p_2,-45))
# ACLR_1st_slope_NN.append(compute_aclr_first_slope(freqs_NN_p_3, psd_NN_p_3,-45))

# # Plot the PSDs
# ##NN:
# plt.figure(figsize=(10, 6))
# #plt.plot(freqs_NN_no_imp, 10 * np.log10(psd_NN_no_imp), label="E2N - no impairment")
# plt.plot(freqs_NN_p_1, 10 * np.log10(psd_NN_p_1), label="E2N - p=1")
# # plt.plot(freqs_NN_p_2, 10 * np.log10(psd_NN_p_2), label="E2E - p=2")
# plt.plot(freqs_NN_p_3, 10 * np.log10(psd_NN_p_3), label="E2E - p=3")
# #Baseline:
# #lt.plot(freqs_Baseline_no_imp, 10 * np.log10(psd_Baseline_no_imp), label="BL - no impairment")
# plt.plot(freqs_Baseline_p_1, 10 * np.log10(psd_Baseline_p_1), label="BL - p=1")
# # plt.plot(freqs_Baseline_p_2, 10 * np.log10(psd_Baseline_p_2), label="BL - p=2")
# plt.plot(freqs_Baseline_p_3, 10 * np.log10(psd_Baseline_p_3), label="BL - p=3")
# plt.title("Power Spectral Density with RAPP Impairments")
# plt.xlabel("Normalized Frequency")
# plt.xlim([freqs_NN_no_imp[0], freqs_NN_no_imp[-1]])
# plt.ylabel(r"$\mathbb{E}\left[|X(f)|^2\right]$ (dB)")
# plt.ylim(ylim)
# plt.grid(True, which="both")
# plt.legend()
# plt.savefig("PSD_new_P1P3.png")
# plt.show()


# # Plot 1: ACLR_3dB_BL and ACLR_3dB_NN
# plt.figure(figsize=(8, 6))
# plt.plot(smoothness_factors, ACLR_3dB_BL, marker='o', label="ACLR 3dB Baseline")
# plt.plot(smoothness_factors, ACLR_3dB_NN, marker='o', label="ACLR 3dB NN")
# plt.title("ACLR 3dB vs Smoothness Factor")
# plt.xlabel("Smoothness Factor (p)")
# plt.ylabel("ACLR (dB)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 2: ACLR_Sionna_BL and ACLR_Sionna_NN
# plt.figure(figsize=(8, 6))
# plt.plot(smoothness_factors, ACLR_Sionna_BL, marker='o', label="ACLR Sionna Baseline")
# plt.plot(smoothness_factors, ACLR_Sionna_NN, marker='o', label="ACLR Sionna NN")
# plt.title("ACLR Sionna vs Smoothness Factor")
# plt.xlabel("Smoothness Factor (p)")
# plt.ylabel("ACLR (dB)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 3: ACLR_1st_slope_BL and ACLR_1st_slope_NN
# plt.figure(figsize=(8, 6))
# plt.plot(smoothness_factors, ACLR_1st_slope_BL, marker='o', label="ACLR 1st Slope Baseline")
# plt.plot(smoothness_factors, ACLR_1st_slope_NN, marker='o', label="ACLR 1st Slope NN")
# plt.title("ACLR 1st Slope vs Smoothness Factor")
# plt.xlabel("Smoothness Factor (p)")
# plt.ylabel("ACLR (dB)")
# plt.legend()
# plt.grid(True)
# plt.show()