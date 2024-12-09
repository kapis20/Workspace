import numpy as np
import matplotlib.pyplot as plt
import pickle

# Assume `signal` is your RRC-filtered signal shaped as (batch, num_samples)
# Example: Replace this with your actual data
batch_size = 128  # Example batch size
num_samples = 4096  # Number of samples in one signal

# File to save the signals
signal_file = "x_rrcf_signals_no_clipping.pkl"



# Load signals from the file
with open(signal_file, "rb") as f:
    loaded_signals = pickle.load(f)
# Check the loaded data
print("Loaded signals:")
for ebno_db, x_rrcf_signal in loaded_signals.items():
    print(f"EB/N0 = {ebno_db} dB, Signal Shape: {x_rrcf_signal.shape}")
batch_size, num_samples = loaded_signals[8.5].shape  # Assuming you want EB/N0 = 8.5 dB
# Select the signal for processing (batch of signals for EB/N0 = 8.5 dB)
signal_batch = loaded_signals[8.5]  # Shape: (batch_size, num_samples)
print("signal shape:", signal_batch.shape)
# Define parameters
sampling_rate = 7.84e9  # 2*Bandwith
samples_per_symbol = 4  # Oversampling factor
rolloff = 0.3  # Roll-off factor for RRC

# Compute symbol rate
symbol_rate = sampling_rate / samples_per_symbol  # Symbol rate in Hz

# Compute PSD for each signal in the batch
psd_batch = []
for i in range(batch_size):
    # Perform FFT on the signal
    fft_signal = np.fft.fftshift(np.fft.fft(signal_batch[i]))
    psd = 20 * np.log10(np.abs(fft_signal) ** 2)  # Convert power to dB
    psd_batch.append(psd)

# Average PSD across the batch
average_psd = np.mean(psd_batch, axis=0)

# # Compute normalized frequency axis
freqs = np.fft.fftshift(np.fft.fftfreq(num_samples, d=1 / sampling_rate))
# normalized_freqs = freqs / (symbol_rate / 2)  # Normalized frequency (relative to Nyquist)

# Plot PSD
plt.figure(figsize=(10, 6))
plt.plot(freqs, average_psd, label="PSD of RRC Signal")
plt.title("PSD of the Tx Filter (RRC Output)")
plt.xlabel("Normalized Frequency")
plt.ylabel("Power Spectral Density (dB)")
plt.grid(True)
plt.legend()
plt.show()
