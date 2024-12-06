
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
# sampling_rate = 8e9  # Sampling rate of 1 GHz

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





class PhaseNoiseGeneratorTF:
    def __init__(
        self,
        fc=120e9,  # Carrier frequency in Hz
        LBW=187e3,  # Loop bandwidth in Hz
        FOM=[-215, -240, -175, -130],  # Figure of merit values in dB
        fz=[float('inf'), 1e4, 50.3e6, float('inf')],  # Zero-point frequencies in Hz
        P=[10, 20, 20, 20],  # Power consumption in mW
        k=[2, 1, 2, 3],  # Zero-point power factors
        ):
        """
        Initialize the Phase Noise Generator with default or user-specified parameters.

        Parameters:
        fc : float, optional
            Carrier frequency in Hz (default is 120 GHz).
        LBW : float, optional
            Loop bandwidth in Hz (default is 187 kHz).
        FOM : list, optional
            Figure of merit values in dB (default values for 3GPP TR38.803).
        fz : list, optional
            Zero-point frequencies in Hz (default values for 3GPP TR38.803).
        P : list, optional
            Power consumption in mW (default values for 3GPP TR38.803).
        k : list, optional
            Zero-point power factors (default values for 3GPP TR38.803).
        """
        self.fc = tf.constant(fc, dtype=tf.float32)
        self.LBW = tf.constant(LBW, dtype=tf.float32)
        self.FOM = tf.constant(FOM, dtype=tf.float32)
        self.fz = tf.constant(fz, dtype=tf.float32)
        self.P = tf.constant(P, dtype=tf.float32)
        self.k = tf.constant(k, dtype=tf.float32)
        self.PSD0 = self.calculate_psd0()

    def calculate_psd0(self):
        return self.FOM + 20.0 * tf.math.log(self.fc) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) - 10.0 * tf.math.log(self.P) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

    def calculate_psd(self, f, psd0, fz, k):
        """
        Calculate the Power Spectral Density (PSD) for a given frequency range.

        Parameters:
        f : tf.Tensor
            Frequency vector (Hz).
        psd0 : tf.Tensor
            PSD0 value at zero frequency, converted to linear scale (not in dB).
        fz : float
            Zero-point frequency (Hz). If fz = float('inf'), a simplified formula is used.
        k : float
            Slope factor controlling the PSD decay with frequency.

        Returns:
        tf.Tensor
            Calculated PSD for the given frequency vector.
        """
            # Ensure `f` and `k` are of the same dtype
        f = tf.cast(f, tf.float32)
        k = tf.cast(k, tf.float32)
        # Convert PSD0 from dB to linear scale: 10^(psd0/10)
        psd_linear = tf.pow(10.0, psd0 / 10.0)
        # Case 1: If fz is infinite (simplified formula)
        # PSD = PSD_linear / (1 + f^k)
        if fz == float('inf'):
            numerator = psd_linear  # Explicitly assign numerator for clarity
            denominator = 1.0 + tf.pow(f, k)  # Compute denominator
            return numerator / denominator
        # Case 2: If fz is finite (full formula)
        # PSD = PSD_linear * [(1 + (f / fz)^k) / (1 + f^k)]
        else:
            numerator = psd_linear * (1.0 + tf.pow(f / fz, k))  # Compute numerator
            denominator = 1.0 + tf.pow(f, k)  # Compute denominator
            return numerator / denominator

    #@tf.function
    def generate_phase_noise_psd(self, fvec):
        """
        Generate the phase noise PSD for a given frequency vector based on 
        the 3GPP TR38.803 UE model 1.

        Parameters:
        fvec : tf.Tensor
            Frequency vector (Hz) for which the phase noise PSD is computed.

        Returns:
        tf.Tensor
            Phase noise PSD values for the given frequency vector.
        """
        
          # Separate the frequency vector into low (f <= LBW) and high (f > LBW) ranges
        f_low = tf.boolean_mask(fvec, fvec <= tf.cast(self.LBW, fvec.dtype))  # Frequencies below or equal to LBW
        f_high = tf.boolean_mask(fvec, fvec > tf.cast(self.LBW, fvec.dtype))  # Frequencies above LBW
        

        # Compute the PSD for the low-frequency range (f <= LBW)
        # Low-frequency PSD is the sum of S_Ref(f) and S_PLL(f)
        lf_low = (
            self.calculate_psd(f_low, self.PSD0[0], self.fz[0], self.k[0]) +
            self.calculate_psd(f_low, self.PSD0[1], self.fz[1], self.k[1])
        )
        # Compute the PSD for the high-frequency range (f > LBW)
        # High-frequency PSD is the sum of S_VCOv2(f) and S_VCOv3(f)
        lf_high = (
            self.calculate_psd(f_high, self.PSD0[2], self.fz[2], self.k[2]) +
            self.calculate_psd(f_high, self.PSD0[3], self.fz[3], self.k[3])
        )
        
        #combine the low-frequency and high-frequency PSDs into a single tensor
        return tf.concat([lf_low, lf_high], axis=0)

    def generatePhaseNoiseSamples(self, num_samples, sampling_rate):
        num_samples_float = tf.cast(num_samples, tf.float32)
        # Generate the frequency axis
        # f_axis = tf.range(0, num_samples_float) / num_samples_float * sampling_rate
        # f_axis = tf.where(f_axis > sampling_rate / 2, f_axis - sampling_rate, f_axis)
        # f_axis = tf.abs(f_axis)
        f_axis = tf.concat([
            tf.linspace(0.0, sampling_rate / 2.0, num_samples // 2 + 1),  # Positive frequencies
            tf.linspace(-sampling_rate / 2.0, -sampling_rate / num_samples, num_samples // 2)  # Negative frequencies
        ], axis=0)
        f_axis = tf.linspace(-sampling_rate / 2.0, sampling_rate / 2.0, num_samples)

        # Compute the PSD (dB and linear scale)
       
        psd_values = self.generate_phase_noise_psd(tf.abs(f_axis))
      
        psd_linear = tf.pow(tf.constant(10.0, dtype=tf.float32), psd_values / 10.0)

        noise = tf.complex(
            tf.random.normal([num_samples], mean=0.0, stddev=1.0),
            tf.random.normal([num_samples], mean=0.0, stddev=1.0)
        )


        psd_filter = tf.sqrt(psd_linear)
        #psd_filter = psd_filter / tf.sqrt(tf.reduce_sum(psd_filter**2))

        noise_freq_domain = tf.signal.fft(noise)
        filtered_noise_freq = noise_freq_domain * tf.cast(psd_filter, tf.complex64)

        # Transform back to the time domain
        phase_noise = tf.math.real(tf.signal.ifft(filtered_noise_freq))
        phase_noise = phase_noise / tf.sqrt(tf.reduce_mean(phase_noise**2))
        return phase_noise

############################################################
## PSD graphs
# import numpy as np
# # Initialize generators for 120 GHz and 220 GHz
# phase_noise_gen_120GHz = PhaseNoiseGeneratorTF(fc=120e9)
# phase_noise_gen_220GHz = PhaseNoiseGeneratorTF(fc=220e9)

# phase_noise_model_120GHz = PhaseNoise()
# phase_noise_model_220GHz = PhaseNoise(f_carrier= 220e9)
# phase_noise_gen_Ref = PhaseNoise(f_carrier = 20e9)
# frequency_offsets = tf.experimental.numpy.logspace(4, 10, 1000)
# # Compute the PSD for the default frequency range
# #frequency_offsets = phase_noise_model_120GHz.f  # Default frequency range
# psd_values120 = phase_noise_model_120GHz.compute_psd(frequency_offsets)
# psd_values220 = phase_noise_model_220GHz.compute_psd(frequency_offsets)
# psd_calues_Ref = phase_noise_gen_Ref.compute_psd(frequency_offsets)

# #frequency vector for plotting (log-spaced)
# fvec = tf.constant(np.logspace(4, 10, 1000), dtype=tf.float32)  # 10 kHz to 10 GHz

# # Generate PSD values for both 120 GHz and 220 GHz
# lf_120GHz = phase_noise_gen_120GHz.generate_phase_noise_psd(fvec)
# lf_220GHz = phase_noise_gen_220GHz.generate_phase_noise_psd(fvec)

# # Convert PSD to dBc/Hz for plotting
# lf_dBc_Hz_120GHz = 10.0 * tf.math.log(lf_120GHz) / tf.math.log(tf.constant(10.0))
# lf_dBc_Hz_220GHz = 10.0 * tf.math.log(lf_220GHz) / tf.math.log(tf.constant(10.0))

# # Plot PSD vs Frequency Offset
# plt.figure(figsize=(8, 6))
# plt.semilogx(fvec.numpy(), lf_dBc_Hz_120GHz.numpy(), label="3GPP UE model 1 @ 120 GHz", color="purple")
# plt.semilogx(fvec.numpy(), lf_dBc_Hz_220GHz.numpy(), label="3GPP UE model 1 @ 220 GHz", color="green")
# plt.semilogx(frequency_offsets.numpy(), psd_values120.numpy(), label="LMX2595 model @ 120 GHz", color="blue")
# plt.semilogx(frequency_offsets.numpy(), psd_values220.numpy(), label="LMX2595 model @ 220 GHz", color="red")
# plt.semilogx(frequency_offsets.numpy(), psd_calues_Ref.numpy(), label="LMX2595 model @ 20 GHz (ref)", color="orange", linestyle ="--")
# plt.xlabel("Frequency Offset [Hz]")
# plt.ylabel("Phase Noise PSD [dBc/Hz]")
# plt.title("Phase Noise PSD vs Frequency Offset for 120 GHz and 220 GHz")
# plt.grid(which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# plt.show()
# ##################################################################
# # Test parameters
# num_samples = 4096
# sample_rate = 120e6
# signal_tensor = tf.complex(
#     tf.random.normal([num_samples], mean=0.0, stddev=1.0),
#     tf.random.normal([num_samples], mean=0.0, stddev=1.0)
# )
# phase_noise_generator = PhaseNoiseGeneratorTF()

# # Add phase noise to the signal
# noisy_signal = phase_noise_generator.add_PN_to_Sig(signal_tensor, sample_rate, phase_noise_generator)

# # Plot original and noisy signals (real parts only for visualization)
# plt.figure(figsize=(12, 6))
# plt.plot(tf.math.real(signal_tensor).numpy()[:500], label="Original Signal", alpha=0.7)
# plt.plot(tf.math.real(noisy_signal).numpy()[:500], label="Signal with Phase Noise", alpha=0.7, linestyle='--')
# plt.title("Original vs. Phase Noise Impaired Signal")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid()
# plt.show()



# import numpy as np
# # Test parameters
# # Initialize the PhaseNoiseGeneratorTF
# phase_noise_model = PhaseNoiseGeneratorTF()

# # Parameters for generation
# num_samples = 4096  # Large number of samples for good frequency resolution
# sampling_rate = 8e9  # 15 GHz sampling rate

# # Generate phase noise
# signal_tensor = tf.random.normal([num_samples], mean=0.0, stddev=1.0)  # Example signal tensor
# generated_noise = phase_noise_model.add_PN_to_Sig(signal_tensor, sampling_rate, phase_noise_model)

# # Compute FFT and PSD of the generated noise
# freqs = np.fft.fftfreq(num_samples, 1 / sampling_rate)  # Frequency axis
# psd_generated = np.abs(np.fft.fft(generated_noise.numpy()))**2 / num_samples  # Generated PSD

# # Compute theoretical PSD for positive frequencies
# freqs_positive = freqs[:num_samples // 2]  # Positive frequencies
# theoretical_psd = phase_noise_model.generate_phase_noise_psd(tf.constant(freqs_positive, dtype=tf.float32)).numpy()

# # Plot the comparison
# plt.figure(figsize=(10, 6))
# plt.semilogx(freqs_positive, 10 * np.log10(psd_generated[:num_samples // 2]), label="Generated PSD")
# plt.semilogx(freqs_positive, theoretical_psd, label="Theoretical PSD", linestyle="--", color="orange")
# plt.xlabel("Frequency Offset (Hz)")
# plt.ylabel("PSD (dBc/Hz)")
# plt.title("Generated vs Theoretical PSD")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.5)
# plt.show()

# # Print debug information (optional)
# # print("Generated phase noise (standalone):", generated_noise.numpy())
# import numpy as np
# # Parameters for generation
# num_samples = 4028  # Large number of samples for good frequency resolution
# sampling_rate = 8e9  # Sampling rate in Hz
# # Instantiate your class
# phase_noise_gen = PhaseNoiseGeneratorTF()

# # # Generate phase noise
# generated_noise = phase_noise_gen.generatePhaseNoiseSamples(num_samples, sampling_rate)



# # # Compute FFT and PSD
# freqs = np.fft.fftfreq(num_samples, 1/sampling_rate)
# print("Frequencies are ",freqs)
# psd_generated = np.abs(np.fft.fft(generated_noise.numpy()))**2 / num_samples

# # # Compute theoretical PSD for comparison
# freqs_positive = freqs[:num_samples // 2]  # Positive frequencies

# psd_theoretical = phase_noise_gen.generate_phase_noise_psd(freqs_positive).numpy()

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