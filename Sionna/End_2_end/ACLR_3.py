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




# File to save the signals
signal_file = "x_rrcf_signals_no_clipping.pkl"



# Load signals from the file
with open(signal_file, "rb") as f:
    loaded_signals = pickle.load(f)
# Check the loaded data
for ebno_db, x_rrcf_signal in loaded_signals.items():
    print(f"EB/N0 = {ebno_db} dB, Signal Shape: {x_rrcf_signal.shape}")
batch_size, num_samples = loaded_signals[8.5].shape  # Assuming you want EB/N0 = 8.5 dB
# Select the signal for processing (batch of signals for EB/N0 = 8.5 dB)
signal_batch = loaded_signals[8.5]  # Shape: (batch_size, num_samples)


###########################################################################
# Compute FFT for PSD
fft_size = 1024
# Compute PSD for each signal in the batch
psd_batch = []
for i in range(batch_size):
    fft_signal = np.fft.fft(signal_batch[i], n=fft_size)
    fft_signal = np.fft.fftshift(fft_signal)  # Center FFT
    psd = np.abs(fft_signal) ** 2  # Power Spectral Density
    psd_batch.append(psd)
# Normalize frequency axis
# Average PSD across the batch
average_psd = np.mean(psd_batch, axis=0)
freq_axis = np.linspace(-0.5, 0.5, fft_size) * samples_per_symbol

# Plot PSD /normalzied power and frequency 
plt.figure(figsize=(12, 6))
plt.plot(freq_axis, 10 * np.log10(average_psd/np.max(average_psd)), label="PSD (dB)")
plt.title("Power Spectral Density (PSD) of Filtered Signal")
plt.xlabel("Normalized Frequency")
plt.ylabel("Power (dB)")
plt.grid()
plt.legend()
plt.show()

###################################################################
# ACLR calculations 
###################################################################

fft_size = len(freq_axis)  # Assuming freq_axis is from the PSD plot

# Define frequency ranges
in_band_range = (freq_axis >= -(0.5 + beta)) & (freq_axis <= (0.5 + beta))
out_of_band_range = ~in_band_range  # Complement of in-band range

# Compute in-band and out-of-band power
in_band_power = np.sum(average_psd[in_band_range])
out_of_band_power = np.sum(average_psd[out_of_band_range])

# Calculate ACLR
aclr = out_of_band_power / in_band_power

# Display the result
print(f"ACLR (Adjacent Channel Leakage Ratio): {10 * np.log10(aclr):.2f} dB")