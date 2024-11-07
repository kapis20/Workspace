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

# Create the BLER plot
plt.figure(figsize=(10, 8))
#Plot BLER 
# Plot BLER
plt.semilogy(ebno_dbs, BLER, 'o-', c='C0', label='BLER - autoencoder-NN')

# Plot BER
plt.semilogy(ebno_dbs, BER, 'x--', c='C1', label='BER - autoencoder-NN')
# Add labels and grid
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("Error rate (BLER/BER")
plt.grid(which="both", linestyle='--', linewidth=0.5)
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
