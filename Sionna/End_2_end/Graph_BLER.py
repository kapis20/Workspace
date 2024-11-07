import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
results_filename = "bler_results.pkl"
with open(results_filename, 'rb') as f:
    results = pickle.load(f)

# # Check the structure of the loaded data
# print("Data structure:")
# print("ebno_dbs:", type(results['ebno_dbs']), results['ebno_dbs'])
# print("BLER:", type(results['BLER']), results['BLER'])
# print("BER:", type(results['BER']), results['BER'])

# Extract SNR values, BLER, and BER specifically for 'autoencoder-NN'
ebno_dbs = results['ebno_dbs']['autoencoder-NN']
BLER = results['BLER']['autoencoder-NN']
BER = results['BER']['autoencoder-NN']

# Create a figure with two subplots, one for BER and one for BLER
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot BER on the top subplot
ax1.semilogy(ebno_dbs, BER, 'x--', c='C1', label='BER - autoencoder-NN')
ax1.set_ylabel("BER")
ax1.grid(which="both", linestyle='--', linewidth=0.5)
ax1.legend()
ax1.set_ylim((1e-4, 1.0))

# Plot BLER on the bottom subplot
ax2.semilogy(ebno_dbs, BLER, 'o-', c='C0', label='BLER - autoencoder-NN')
ax2.set_xlabel(r"$E_b/N_0$ (dB)")
ax2.set_ylabel("BLER")
ax2.grid(which="both", linestyle='--', linewidth=0.5)
ax2.legend()
ax2.set_ylim((1e-4, 1.0))

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()