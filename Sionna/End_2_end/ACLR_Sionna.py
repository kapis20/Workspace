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
signal_file = "x_rrcf_signals_no_clippingNN_conv_no_imp.pkl"
#signal_file_noisy = "x_rrcf_Rapp.pkl"
signal_file_noisy="x_rrcf_signals_no_clippingNN_conv.pkl"


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



empirical_psd(signal_batch_with_RAPP, show = True, oversampling = 4.0, ylim=(-100,3))