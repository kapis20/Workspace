import numpy as np

class PhaseNoiseGenerator:
    def __init__(self, fc, LBW, FOM, fz, P, k):
        """
        Initialize the Phase Noise Generator.

        Parameters:
        fc : float
            Carrier frequency in Hz.
        LBW : float
            Loop bandwidth in Hz.
        FOM : list
            Figure of merit values in dB for REF clk, PLL, VCO (1/f^2, 1/f^3).
        fz : list
            Zero-point frequencies in Hz.
        P : list
            Power consumption in mW.
        k : list
            Zero-point power factors.
        """
        self.fc = fc
        self.LBW = LBW
        self.FOM = np.array(FOM)
        self.fz = np.array(fz)
        self.P = np.array(P)
        self.k = np.array(k)
        self.PSD0 = self.calculate_psd0()

    def calculate_psd0(self):
        """
        Calculate PSD0 for each component based on FOM, fc, and P.
        """
        return self.FOM + 20 * np.log10(self.fc) - 10 * np.log10(self.P)

    def calculate_psd(self, f, psd0, fz, k):
        """
        Calculate the PSD for a given frequency range, zero-point frequency, and power factor.

        Parameters:
        f : ndarray
            Frequency vector in Hz.
        psd0 : float
            PSD0 value for the component.
        fz : float
            Zero-point frequency.
        k : float
            Zero-point power factor.
        """
        if fz == float('inf'):
            return np.power(10.0, psd0 / 10.0) / (1.0 + np.power(f, k))
        else:
            return np.power(10.0, psd0 / 10.0) * (
                (1.0 + np.power(f / fz, k)) / (1.0 + np.power(f, k))
            )

    def generate_phase_noise_psd(self, fvec):
        """
        Generate the phase noise PSD for the given frequency vector.

        Parameters:
        fvec : ndarray
            Frequency vector in Hz.

        Returns:
        lf : ndarray
            Phase noise PSD values for the given frequencies.
        """
        f_low = fvec[fvec <= self.LBW]
        f_high = fvec[fvec > self.LBW]

        # Calculate PSD for low and high frequency ranges
        lf_low = (
            self.calculate_psd(f_low, self.PSD0[0], self.fz[0], self.k[0]) +
            self.calculate_psd(f_low, self.PSD0[1], self.fz[1], self.k[1])
        )
        lf_high = (
            self.calculate_psd(f_high, self.PSD0[2], self.fz[2], self.k[2]) +
            self.calculate_psd(f_high, self.PSD0[3], self.fz[3], self.k[3])
        )

        return np.concatenate([lf_low, lf_high])

# Example usage
fc = 120e9  # Carrier frequency in Hz
LBW = 187e3  # Loop bandwidth in Hz
FOM = [-215, -240, -175, -130]  # Figure of merit values in dB
fz = [float('inf'), 1e4, 50.3e6, float('inf')]  # Zero-point frequencies in Hz
P = [10, 20, 20, 20]  # Power consumption in mW
k = [2, 1, 2, 3]  # Zero-point power factors

# Frequency vector for PSD calculation
fvec = np.logspace(3, 10, 1000)  # Frequencies from 1 kHz to 10 GHz

# Generate phase noise
phase_noise_gen = PhaseNoiseGenerator(fc, LBW, FOM, fz, P, k)
lf = phase_noise_gen.generate_phase_noise_psd(fvec)

# Print or plot the results
print("Generated Phase Noise PSD:", lf)
