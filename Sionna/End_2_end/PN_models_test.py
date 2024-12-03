############ PN Transmit ##############

import numpy as np
import matplotlib.pyplot as plt

# Parameters from Table II
PSD0_dB = -72  #dBc/Hz 
PSD0 = 10**(PSD0_dB / 10)  # Convert to linear scale
# print("PSD0 linear is", PSD0)
f_carrier = 120e9  # Carrier frequency (120 GHz)
f_ref = 20e9  # Reference frequency (20 GHz)
scale_factor = 20 * np.log10(f_carrier / f_ref)  # Scaling for 120 GHz

# Zero and pole frequencies (from Table II)
fz = [3e4, 1.75e7]  # Zero frequencies (Hz)
fp = [10, 3e5]  # Pole frequencies (Hz)

# Exponents (from Table II)
alpha_zn = [1.4, 2.55]  # Powers for zeros
alpha_pn = [1.0, 2.95]  # Powers for poles

# Frequency axis, frequency offset 
f = np.logspace(2, 8, 1000)  # Offset frequencies (100 Hz to 100 MHz)

# Phase noise PSD calculation
numerator = np.prod([1 + (f / fz_i)**alpha for fz_i, alpha in zip(fz, alpha_zn)], axis=0)
denominator = np.prod([1 + (f / fp_i)**alpha for fp_i, alpha in zip(fp, alpha_pn)], axis=0)
psd = PSD0 * (numerator / denominator)

# print("Numerator shape:", numerator.shape, "Sample values:", numerator[:5])
# print("Denominator shape:", denominator.shape, "Sample values:", denominator[:5])
# print("PSD (linear): Sample values:", psd[:5])
# Scale for carrier frequency
psd_db = 10 * np.log10(psd) + scale_factor  # Convert to dB scale
#print("PSD (dBc/Hz): Sample values:", psd_db[:5])
# Plot the PSD
plt.figure(figsize=(10, 6))
plt.semilogx(f, psd_db)
plt.xlabel("Frequency Offset (Hz)")
plt.ylabel("Phase Noise PSD (dBc/Hz)")
plt.title("Transmitter Phase Noise PSD for 120 GHz")
plt.grid(True, which = 'major')
plt.grid(True, which = 'minor')
plt.show()


