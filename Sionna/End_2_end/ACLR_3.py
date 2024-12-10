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
signal_file_noisy = "x_rrcf_Rapp.pkl"


# Load signals from the file
with open(signal_file, "rb") as f:
    loaded_signals = pickle.load(f)


# Load signals from the file
with open(signal_file_noisy, "rb") as f:
    loaded_signals_noisy = pickle.load(f)

# Initialize CCDFCalculator    
# Check the loaded data
for ebno_db, x_rrcf_signal in loaded_signals.items():
    print(f"EB/N0 = {ebno_db} dB, Signal Shape: {x_rrcf_signal.shape}")
batch_size, num_samples = loaded_signals[9].shape  # Assuming you want EB/N0 = 8.5 dB
signal_batch = loaded_signals[9]  # Shape: (batch_size, num_samples)

for ebno_db, x_rrcf_signal in loaded_signals_noisy.items():
    print(f"EB/N0 = {ebno_db} dB, Signal Shape (with Noise): {x_rrcf_signal.shape}")
batch_size_RAPP, num_samples = loaded_signals_noisy[9].shape  # Assuming you want EB/N0 = 8.5 dB
signal_batch_with_RAPP = loaded_signals_noisy[9]  # With RAPP


###########################################################################
# Compute FFT for PSD
fft_size = 5712 #at least 2x signal lenght (my signal is (128,2856))
# Compute PSD for each signal in the batch#
psd_batch = []

for i in range(batch_size):
    fft_signal = np.fft.fft(signal_batch[i], n=fft_size)
    fft_signal = np.fft.fftshift(fft_signal)  # Center FFT
    #psd = np.abs(fft_signal) ** 2  # Power Spectral Density
    psd = (np.abs(fft_signal) ** 2) / fft_size  # Normalize power
    psd_batch.append(psd)
# Average PSD across the batch in linear scale
average_psd = np.mean(psd_batch, axis=0)
# Compute PSD for the signal batch with RAPP
psd_batch_noisy = []
for i in range(batch_size_RAPP):
    fft_signal_noisy = np.fft.fft(signal_batch_with_RAPP[i], n=fft_size)
    fft_signal_noisy = np.fft.fftshift(fft_signal_noisy)  # Center FFT
    #psd_noisy = np.abs(fft_signal_noisy) ** 2  # Power Spectral Density
    psd_noisy = (np.abs(fft_signal_noisy) ** 2) / fft_size  # Normalize power
   
    psd_batch_noisy.append(psd)
    
 # Average PSD across the batch in linear scale
average_psd_noisy = np.mean(psd_batch_noisy, axis=0)   
# Normalize frequency axis
freq_axis = np.linspace(-samples_per_symbol/2,samples_per_symbol/2,fft_size)
# Plot PSD of individual signals
# for i in range(min(4, batch_size)):  # Plot first 5 signals
#     fft_signal = np.fft.fft(signal_batch[i], n=fft_size)# / num_samples
#     ft_signal_noisy = np.fft.fft(signal_batch_with_RAPP[i], n=fft_size)
#     fft_signal_noisy = np.fft.fftshift(fft_signal_noisy)  # Center FFT
#     fft_signal = np.fft.fftshift(fft_signal)
#     psd = (np.abs(fft_signal) ** 2) / fft_size
#     psd_noisy = (np.abs(fft_signal_noisy) ** 2) / fft_size  # Normalize power
#     plt.plot(freq_axis, 10 * np.log10(psd), label=f"Signal {i+1}")
#     plt.plot(freq_axis, 10 * np.log10(psd_noisy), label=f"NoisySignal {i+1}")

# plt.title("PSD of Individual Signals")
# plt.xlabel("Normalized Frequency")
# plt.ylabel("Power (dB)")
# plt.legend()
# plt.grid()
# plt.show()



# Plot PSD /normalzied power and frequency 
plt.figure(figsize=(12, 6))
#plt.plot(freq_axis, 10 * np.log10(average_psd/np.max(average_psd)), label="64 QAM, $\\beta$ = 0.3")
plt.plot(freq_axis, 10 * np.log10(average_psd), label="64 QAM, $\\beta$ = 0.3")
plt.plot(
    freq_axis,
    10 * np.log10(average_psd_noisy),
    #10 * np.log10(average_psd_noisy / np.max(average_psd_noisy)),
    linestyle="--",
    color ="red",
    label="Noisy Signal with RAPP"
)
    
plt.title("Power Spectral Density (PSD) of original signal vs RAPP distorted signal")
plt.xlabel("Normalized Frequency")
plt.ylabel("Power (dB)")
plt.grid()
plt.legend()
plt.savefig("PSD_PLOT_ACLR.png",dpi =300)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(signal_batch[0], label="Original Signal")
plt.plot(signal_batch_with_RAPP[0], label="Noisy Signal with RAPP")
plt.title("Signal Comparison (Time Domain)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

original_power = np.mean(np.abs(signal_batch)**2)
rapp_power = np.mean(np.abs(signal_batch_with_RAPP)**2)

print(f"Original Signal Power: {original_power}")
print(f"RAPP Signal Power: {rapp_power}")

# Define adjacent channel mask (frequencies outside ±0.5)
adjacent_channel_mask = (freq_axis < -0.5-beta) | (freq_axis > 0.5+beta)

# Calculate adjacent channel power
adjacent_power_original = np.sum(average_psd[adjacent_channel_mask])
adjacent_power_rapp = np.sum(average_psd_noisy[adjacent_channel_mask])

print(f"Adjacent Channel Power (Original): {10 * np.log10(adjacent_power_original)} dB")
print(f"Adjacent Channel Power (RAPP): {10 * np.log10(adjacent_power_rapp)} dB")

powerInband = np.sum(average_psd[~adjacent_channel_mask])
powerInnadRapp = np.sum(average_psd_noisy[~adjacent_channel_mask])

print(f" Channel Power (Original): {10 * np.log10(powerInband)} dB")
print(f" Channel Power (RAPP): {10 * np.log10(powerInnadRapp)} dB")
###################################################################
# ACLR calculations 
###################################################################

fft_size = len(freq_axis)  # Assuming freq_axis is from the PSD plot

# Define frequency ranges
in_band_range = (freq_axis >= -(0.5 + beta)) & (freq_axis <= (0.5 + beta))
#in_band_range = (freq_axis >= -(0.5)) & (freq_axis <= (0.5))
out_of_band_range = ~in_band_range  # Complement of in-band range

# Compute in-band and out-of-band power
in_band_power = np.sum(average_psd[in_band_range])
out_of_band_power = np.sum(average_psd[out_of_band_range])

# Compute in-band and out-of-band power noisy 
in_band_power_noisy = np.sum(average_psd_noisy[in_band_range])
out_of_band_power_noisy = np.sum(average_psd_noisy[out_of_band_range])

# Calculate ACLR
aclr = out_of_band_power / in_band_power
aclr_noisy = out_of_band_power_noisy/in_band_power_noisy

# Display the result
print(f"ACLR (Adjacent Channel Leakage Ratio): {10 * np.log10(aclr):.2f} dB")
print(f"ACLR Noisy (Adjacent Channel Leakage Ratio): {10 * np.log10(aclr_noisy):.2f} dB")