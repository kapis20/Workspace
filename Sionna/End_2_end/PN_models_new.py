
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
        #Scaling log 10 base but using ln in tf
        self.scale_factor = 20 * (tf.math.log(f_carrier / f_ref) / tf.math.log(10.0)) 
        #print("Scale factor dtype:", self.scale_factor.dtype)


                                        
        
        # Default zero and pole frequencies and exponents (if not provided)
        self.fz = fz if fz is not None else [3e4, 1.75e7]  # Zero frequencies (Hz)
        self.fp = fp if fp is not None else [10, 3e5]       # Pole frequencies (Hz)
        self.alpha_zn = alpha_zn if alpha_zn is not None else [1.4, 2.55]  # Zero exponents
        self.alpha_pn = alpha_pn if alpha_pn is not None else [1.0, 2.95]  # Pole exponents
        self.f = f if f is not None else tf.experimental.numpy.logspace(2, 10, 1000)  # From 100 Hz to 10000 MHz (log scale)
    def compute_psd(self, f):
        """
        Compute the phase noise PSD for a given frequency offset.
        :param f: Frequency offset array (in Hz).
        :return: Phase noise PSD in dB/Hz.
        """
        f = tf.cast(f, dtype=tf.float64)  # Ensure f is float64
        numerator = tf.reduce_prod([1 + (f / fz_i)**alpha for fz_i, alpha in zip(self.fz, self.alpha_zn)], axis=0)
        denominator = tf.reduce_prod([1 + (f / fp_i)**alpha for fp_i, alpha in zip(self.fp, self.alpha_pn)], axis=0)
        psd = self.PSD0 * (numerator / denominator)
         # Cast scale factor to float64 to match psd
        scale_factor = tf.cast(self.scale_factor, dtype=tf.float64)
        #print("PSD dtype:", psd.dtype)
        return 10 * (tf.math.log(psd) / tf.math.log(tf.constant(10.0, dtype=tf.float64))) + scale_factor


    def generate_phase_noise(self, num_samples, sampling_rate):
        """
        Generate phase noise samples based on the PSD.
        :param num_samples: Number of time-domain samples to generate.
        :param sampling_rate: Sampling rate in Hz.
        :return: Tensor of time-domain phase noise samples.
        """
        #  # Convert num_samples to an integer if it's a TensorFlow tensor
        # if isinstance(num_samples, tf.Tensor):
        #     num_samples = int(num_samples.numpy())
        # Ensure num_samples is a TensorFlow tensor
        # num_samples = tf.cast(num_samples, tf.int32)
        # num_samples_int = tf.get_static_value(num_samples)  # Get static value for NumPy compatibility
        # # Frequency axis
        # f_axis = np.fft.fftfreq(num_samples_int, d=1/sampling_rate)  # Frequency bins
        # f_axis = np.abs(f_axis)  # Consider positive frequencies only for PSD
         # Generate frequency axis in TensorFlow
        num_samples_float = tf.cast(num_samples, tf.float32)
        f_axis = tf.range(0, num_samples_float) / num_samples_float * sampling_rate
        f_axis = tf.where(f_axis > sampling_rate / 2, f_axis - sampling_rate, f_axis)  # Wrap negative frequencies
        f_axis = tf.abs(f_axis)  # Consider positive frequencies only for PSD
        # Compute PSD values for the frequency axis
        psd_linear = tf.pow(tf.constant(10.0, dtype=tf.float64), self.compute_psd(f_axis) / 10.0)
        #psd_linear = 10**(self.compute_psd(f_axis) / 10)  # Convert PSD to linear scale
        
        # Generate white Gaussian noise in the frequency domain
        noise = tf.complex(
            tf.random.normal([num_samples], mean=0.0, stddev=1.0),
            tf.random.normal([num_samples], mean=0.0, stddev=1.0)
        )
        
        # Apply the square root of the PSD as a filter
        # Normalize the PSD filter
        #psd_filter = tf.sqrt(psd_linear / tf.reduce_sum(psd_linear))

        psd_filter = tf.sqrt(psd_linear)
        noise_freq_domain = tf.signal.fft(noise)
        filtered_noise_freq = noise_freq_domain * tf.cast(psd_filter, tf.complex64)
        
        # Transform back to time domain
        phase_noise = tf.signal.ifft(filtered_noise_freq)
        return tf.math.real(phase_noise)  # Return real part
# # Instantiate the PhaseNoise class with default parameters
phase_noise_model = PhaseNoise()

# # Compute the PSD for the default frequency range
# frequency_offsets = phase_noise.f  # Default frequency range
# psd_values = phase_noise.compute_psd(frequency_offsets)

# # Plot the Phase Noise PSD vs Frequency Offset
# plt.figure(figsize=(10, 6))
# plt.plot(frequency_offsets.numpy(), psd_values.numpy(), label="Phase Noise PSD")
# plt.xscale('log')  # Logarithmic scale for frequency axis
# plt.xlabel("Frequency Offset (Hz)")
# plt.ylabel("PSD (dB/Hz)")
# plt.title("Phase Noise PSD vs Frequency Offset")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# # plt.show()


# import numpy as np
# # # Parameters for generation
# num_samples = 4028  # Large number of samples for good frequency resolution
# sampling_rate = 15000000000  # Sampling rate of 1 GHz

# # Generate phase noise
# generated_noise = phase_noise_model.generate_phase_noise(num_samples, sampling_rate)

# # Compute FFT and PSD
# freqs = np.fft.fftfreq(num_samples, 1/sampling_rate)
# psd_generated = np.abs(np.fft.fft(generated_noise.numpy()))**2 / num_samples

# # Compute theoretical PSD for comparison
# freqs_positive = freqs[:num_samples // 2]  # Positive frequencies
# psd_theoretical = phase_noise_model.compute_psd(freqs_positive).numpy()

# # Plot comparison
# plt.figure(figsize=(10, 6))
# plt.semilogx(freqs_positive, 10 * np.log10(psd_generated[:num_samples // 2]), label="Generated PSD")
# plt.semilogx(freqs_positive, psd_theoretical, label="Theoretical PSD", linestyle="--")
# plt.xlabel("Frequency Offset (Hz)")
# plt.ylabel("PSD (dBc/Hz)")
# plt.title("Generated vs Theoretical PSD")
# plt.legend()
# plt.grid()
# plt.show()

# print("Generated phase noise (standalone):", generated_noise.numpy())
# print("PhaseNoise parameters (standalone):", vars(phase_noise_model))