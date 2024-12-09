import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fftshift

# Example signal: Replace this with your signal of shape (batch, num_symbols)
batch_size = 5
num_symbols = 1024
sampling_frequency = 1000  # Replace with your actual sampling frequency (Hz)

# Generate a sample signal for demonstration (e.g., sine wave + noise)
np.random.seed(0)  # For reproducibility
signals = np.sin(2 * np.pi * 50 * np.linspace(0, 1, num_symbols)) + 0.5 * np.random.randn(batch_size, num_symbols)

# Compute PSD using Welch's method for each signal in the batch
psds = []
frequencies = None

for signal in signals:
    freq, psd = welch(signal, fs=sampling_frequency)
    psds.append(psd)
    frequencies = freq  # All frequencies are the same for each batch

# Convert to NumPy array for further processing if needed
psds = np.array(psds)

# Normalize frequencies to -0.5 to 0.5
frequencies_normalized = freq / sampling_frequency  # Normalize to 0-0.5
frequencies_two_sided = fftshift(np.concatenate((-frequencies_normalized[::-1], frequencies_normalized[1:])))
psds_two_sided = fftshift(np.concatenate((psds[:, ::-1], psds[:, 1:]), axis=1))

# Plot PSD for the first signal in the batch
plt.figure(figsize=(10, 6))
plt.plot(frequencies_two_sided, psds_two_sided[0], label='Batch 0 PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Power/Hz)')
plt.title('Power Spectral Density (Welch Method)')
plt.grid(True)
plt.legend()
plt.show()
