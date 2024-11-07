import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
results_filename = "bler_results.pkl"
with open(results_filename, 'rb') as f:
    ebno_dbs, BLER = pickle.load(f)

# Create the BLER plot
plt.figure(figsize=(10, 8))
plt.semilogy(ebno_dbs, BLER['autoencoder-NN'], 'o-', c='C0', label='autoencoder-NN')

# Add labels and grid
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both", linestyle='--', linewidth=0.5)
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
