import pickle
import matplotlib.pyplot as plt
import numpy as np

##############################################
#PLot BER and BLER function
##############################################
def plot_ber_bler(results_filename):
# Load the data from the file

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


###################################################
#Constellation function
###################################################
def plot_constellation(constellation_data_filename, stage="constellation_before", num_bits_per_symbol=6):
    # Load the constellation data from the file
    with open(constellation_data_filename, "rb") as f:
        constellation_data = pickle.load(f)
    
    # Check that the stage exists in the constellation_data
    if stage not in constellation_data:
        raise ValueError(f"Stage '{stage}' not found in constellation data. Available stages: {list(constellation_data.keys())}")
    
    # Extract the constellation points for the specified stage
    points = constellation_data[stage]
    real = points.real
    imag = points.imag
    
    # Plot the constellation points
    plt.figure(figsize=(6, 6))
    plt.scatter(real, imag, label=f"Constellation Points ({stage})", color='orange')
    
    # Annotate each point with its binary representation
    for i, point in enumerate(points):
        plt.text(point.real, point.imag, f'{i:0{num_bits_per_symbol}b}', 
                 ha='center', va='center', fontsize=8)
    
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title(f"Constellation Plot ({stage})")
    plt.grid(True)
    plt.legend()
    plt.show()



##################################################
#calling functions 
##################################################
# Plot the BER and BLER results from "bler_results.pkl"
plot_ber_bler("bler_results.pkl")

# Plot the constellation before training
plot_constellation("constellation_data.pkl", stage="constellation_before", num_bits_per_symbol=6)

# Plot the constellation after training
plot_constellation("constellation_data.pkl", stage="constellation_after", num_bits_per_symbol=6)