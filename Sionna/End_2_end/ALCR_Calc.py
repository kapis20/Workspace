import numpy as np
import pickle
from scipy.signal import welch
from scipy.integrate import simpson


class ACLRCalculator:
    def __init__(self, signal, sampling_frequency, bandwidth, excess_bandwidth, signal_batch):
        """
        Initialize the ACLRCalculator class.

        Args:
            signal (numpy array): The signal coming out of the transmit filter.
            sampling_frequency (float): Sampling frequency of the signal (Hz).
            bandwidth (float): Allocated bandwidth (Hz).
            excess_bandwidth (float): Excess bandwidth factor (e.g., 0.25 for 25%).
        """
        self.signal = signal
        self.fs = sampling_frequency
        self.bandwidth = bandwidth
        self.excess_bandwidth = excess_bandwidth
        self.batch = signal_batch

    def compute_psd(self, nperseg=1024):
        """
        Compute the Power Spectral Density (PSD) of the signal using Welch's method.

        Args:
            nperseg (int): Number of points per segment for the Welch method.

        Returns:
            freqs (numpy array): Frequency values.
            psd (numpy array): Power Spectral Density values.
        """
        aclr_values = []
        
        for signal in self.batch:
            # Compute Power Spectral Density (PSD) for each signal in the batch
            freqs, psd = welch(self.signal, self.fs, nperseg=1024)

            # Define frequency ranges
            in_band = (freqs >= 0) & (freqs <= 0.5 * self.fs)  # Main channel
            out_of_band = freqs > (0.5 + self.excess_bandwidth) * self.fs  # Adjacent channel
            print("in badn is", in_band)
            print("in out_of_band is", out_of_band)
            # Calculate power in each region
            P_main = np.sum(psd[in_band])
            P_adj = np.sum(psd[out_of_band])

            # Calculate ACLR in dB
            aclr = 10 * np.log10(P_main / P_adj)
            aclr_values.append(aclr)
            return freqs, psd
    
    # def compute_aclr(self, nperseg=1024):
    #     """
    #     Compute the Adjacent Channel Leakage Ratio (ACLR).

    #     Args:
    #         nperseg (int): Number of points per segment for the Welch method.

    #     Returns:
    #         aclr (float): The ACLR value.
    #     """
    #     freqs, psd = self.compute_psd(nperseg=nperseg)

    #      # Normalize frequencies to be between -0.5 and 0.5
    #     freqs = np.fft.fftshift(freqs)  # Shift zero frequency to the center
    #     print("Fregs are", freqs)
    #     freqs = freqs/self.fs
    #     print("Fregs are", freqs)
       
    #     # Define channel bounds
    #     main_channel_lower = -self.bandwidth / (2*self.fs)
    #     main_channel_upper = self.bandwidth / (2*self.fs)
    #     print("main_channel_lower is",main_channel_lower)
    #     print("main_channel_upper is",main_channel_upper)
    #     adjacent_channel_lower = main_channel_upper
    #     adjacent_channel_upper = main_channel_upper + self.bandwidth * (1 + self.excess_bandwidth)
    #     print("adjacent_channel_lower is",adjacent_channel_lower)
    #     print("adjacent_channel_upper is",adjacent_channel_upper)
    #     # Convert bounds to indices
    #     main_channel_indices = (freqs >= main_channel_lower) & (freqs <= main_channel_upper)
    #     adjacent_channel_indices = (freqs > adjacent_channel_lower) & (freqs <= adjacent_channel_upper)

    #     # Integrate power over main and adjacent channels
    #     main_channel_power = simpson(psd[main_channel_indices], freqs[main_channel_indices])
    #     adjacent_channel_power = simpson(psd[adjacent_channel_indices], freqs[adjacent_channel_indices])

    #     # Calculate ACLR
    #     aclr = 10 * np.log10(main_channel_power / adjacent_channel_power)
    #     return aclr

 

# File to save the signals
signal_file = "x_rrcf_signals_no_clipping.pkl"



# Load signals from the file
with open(signal_file, "rb") as f:
    loaded_signals = pickle.load(f)


# Define parameters
num_bits_per_symbol = 6 # Baseline is 64-QAM
#modulation_order = 2**num_bits_per_symbol
coderate = 0.75 #0.75 # Coderate for the outer code
n = 4092 #4098 #4096 Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
bandwidth_allocated = 3.93e9  # Allocated bandwidth from the paper (in Hz)
#filter
beta = 0.3 # Roll-off factor
span_in_symbols = 32 # Filter span in symbold
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor

bandwidth = num_symbols_per_codeword / span_in_symbols  # Bandwidth calculation based on symbol rate
sampling_frequency = samples_per_symbol * bandwidth  # Sampling frequency based on oversampling factor

# # Calculate symbol rate from allocated bandwidth
# symbol_rate = bandwidth_allocated / (1 + beta)
# # Calculate sampling frequency
# sampling_frequency = symbol_rate * samples_per_symbol



# Calculate ACLR for each signal in the loaded dataset
for ebno_db, x_rrcf_signal in loaded_signals.items():
    aclr_calculator = ACLRCalculator(x_rrcf_signal, sampling_frequency, bandwidth_allocated, beta,10)
    aclr_value = aclr_calculator.compute_aclr()
    print(f"ACLR: {aclr_value:.2f} dB")



