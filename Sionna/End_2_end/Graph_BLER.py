import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
results_filename = "awgn_autoencoder_results"
with open(results_filename, 'rb') as f:
    ebno_dbs, BLER = pickle.load(f)

# Create the BLER plot
plt.figure(figsize=(10, 8))
plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c='C0', label='Baseline')

# Add labels and grid
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both", linestyle='--', linewidth=0.5)
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
