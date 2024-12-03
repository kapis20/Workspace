import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt 

class PhaseNoise:
    def __init__(self, PSD0_dB=-72, f_carrier=120e9, f_ref=20e9, f=None, fz=None, fp=None, alpha_zn=None, alpha_pn=None):
        """
        Initialize phase noise parameters.
        :param PSD0_dB: Reference phase noise power at zero frequency (in dBc/Hz).
        :param f_carrier: Carrier frequency (e.g., 120 GHz).
        :param f_ref: Reference frequency (e.g., 20 GHz).
        :param fz: List of zero frequencies.
        :param fp: List of pole frequencies.
        :param alpha_zn: List of exponents for zero frequencies.
        :param alpha_pn: List of exponents for pole frequencies.
        """
        self.PSD0_dB = PSD0_dB
        self.PSD0 = 10**(PSD0_dB / 10)  # Convert to linear scale
        self.f_carrier = f_carrier
        self.f_ref = f_ref
        self.scale_factor = 20 * np.log10(f_carrier / f_ref)  # Scaling for carrier frequency
        
        # Default zero and pole frequencies and exponents (if not provided)
        self.fz = fz if fz is not None else [3e4, 1.75e7]  # Zero frequencies (Hz)
        self.fp = fp if fp is not None else [10, 3e5]       # Pole frequencies (Hz)
        self.alpha_zn = alpha_zn if alpha_zn is not None else [1.4, 2.55]  # Zero exponents
        self.alpha_pn = alpha_pn if alpha_pn is not None else [1.0, 2.95]  # Pole exponents
        self.f = f if f is not None else np.logspace(2, 8, 1000)  # From 100 Hz to 100 MHz (log scale)
    def compute_psd(self, f):
        """
        Compute the phase noise PSD for a given frequency offset.
        :param f: Frequency offset array (in Hz).
        :return: Phase noise PSD in dB/Hz.
        """
        numerator = np.prod([1 + (f / fz_i)**alpha for fz_i, alpha in zip(self.fz, self.alpha_zn)], axis=0)
        denominator = np.prod([1 + (f / fp_i)**alpha for fp_i, alpha in zip(self.fp, self.alpha_pn)], axis=0)
        psd = self.PSD0 * (numerator / denominator)
        return 10 * np.log10(psd) + self.scale_factor

    def generate_phase_noise(self, num_samples, sampling_rate):
        """
        Generate phase noise samples based on the PSD.
        :param num_samples: Number of samples to generate.
        :param sampling_rate: Sampling rate of the signal (in Hz).
        :return: Phase noise samples (in radians).
        """
        f = np.fft.fftfreq(num_samples, 1 / sampling_rate)[:num_samples // 2]
        psd = self.compute_psd(f)  # Compute PSD for positive frequencies
        psd_linear = 10**(psd / 10)  # Convert to linear scale
        
        # Generate random Gaussian noise in the frequency domain
        noise_amplitude = np.sqrt(psd_linear)
        random_phase = np.random.normal(0, 1, len(f)) + 1j * np.random.normal(0, 1, len(f))
        noise_spectrum = noise_amplitude * random_phase

        # Symmetrize to get a real signal in time domain
        full_spectrum = np.zeros(num_samples, dtype=complex)
        full_spectrum[:num_samples // 2] = noise_spectrum
        full_spectrum[num_samples // 2:] = np.conj(noise_spectrum[::-1])

        # Convert to time domain and return phase noise in radians
        phase_noise = np.fft.ifft(full_spectrum).real
        return phase_noise 

    # def add_phase_noise(self, signal_tensor, sampling_rate):
    #     """
    #     Add phase noise to a signal tensor.
    #     :param signal_tensor: Signal (complex tensor) to which phase noise is added.
    #     :param sampling_rate: Sampling rate of the signal (in Hz).
    #     :return: Signal with phase noise added.
    #     """
    #     num_samples = signal_tensor.shape[0]
    #     phase_noise = self.generate_phase_noise(num_samples, sampling_rate)
        
    #     # Add phase noise as a complex exponential
    #     phase_noise_tensor = tf.constant(np.exp(1j * phase_noise), dtype=signal_tensor.dtype)
    #     return signal_tensor * phase_noise_tensor


phase_noise_model = PhaseNoise()#default parameters
# Frequency offset range for PSD computation
f = np.logspace(2, 8, 1000)  # From 100 Hz to 100 MHz (log scale)

# Compute PSD for these frequency offsets
psd_db = phase_noise_model.compute_psd(f)

# Plot the PSD vs Frequency Offset
plt.figure(figsize=(10, 6))
plt.semilogx(f, psd_db, label="120 GHz Transmitter Phase Noise")
plt.xlabel("Frequency Offset (Hz)")
plt.ylabel("Phase Noise PSD (dBc/Hz)")
plt.title("Transmitter Phase Noise PSD for 120 GHz")
plt.grid(True, which="major", linestyle='-')
plt.grid(True, which="minor", linestyle=':')
plt.legend()
plt.show()

