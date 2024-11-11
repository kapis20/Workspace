import pickle
import matplotlib.pyplot as plt
import numpy as np


#############################################
# names of the files
#############################################
loss_file_path = "loss_values.pkl"

##############################################
#PLot BER and BLER function
##############################################
def plot_ber_bler(results_filename,baseline_filename):
# Load the data from the file

    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    with open(baseline_filename, 'rb') as f:
        baseline_results = pickle.load(f)

    # Extract EB/No values, BLER, and BER specifically for 'autoencoder-NN'
    ebno_dbs_nn = results['ebno_dbs']['autoencoder-NN']
    BLER_nn = results['BLER']['autoencoder-NN']
    BER_nn = results['BER']['autoencoder-NN']

    ebno_dbs_baseline = baseline_results['ebno_dbs']['baseline']
    BLER_baseline = baseline_results['BLER']['baseline']
    BER_baseline = baseline_results['BER']['baseline']

    # Create a figure with two subplots, one for BER and one for BLER
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot BER on the top subplot
     # Plot BER on the top subplot
    ax1.semilogy(ebno_dbs_nn, BER_nn, 'x--', c='C1', label='BER - autoencoder-NN')
    ax1.semilogy(ebno_dbs_baseline, BER_baseline, 'x--', c='C2', label='BER - Baseline')
    ax1.set_ylabel("BER")
    ax1.grid(which="both", linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_ylim((1e-4, 1.0))

    # Plot BLER on the bottom subplot
    ax2.semilogy(ebno_dbs_nn, BLER_nn, 'o-', c='C0', label='BLER - autoencoder-NN')
    ax2.semilogy(ebno_dbs_baseline, BLER_baseline, 'o-', c='C3', label='BLER - Baseline')
    ax2.set_xlabel(r"$E_b/N_0$ (dB)")
    ax2.set_ylabel("BLER")
    ax2.grid(which="both", linestyle='--', linewidth=0.5)
    ax2.legend()
    ax2.set_ylim((1e-4, 1.0))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("ber_bler_combined_plot.png")
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
    # Save the plot as an image file
    plt.savefig(f"{stage}_constellation_plot.png")
    plt.show()


def plot_constellation_baseline(baseline_constellation_filename):
    # Load the baseline constellation data
    with open(baseline_constellation_filename, "rb") as f:
        constellation_baseline = pickle.load(f)

    # Plot the baseline constellation
    plt.figure(figsize=(6, 6))
    for ebno, points in constellation_baseline.items():
        real = points.real
        imag = points.imag
        plt.scatter(real, imag, label=f"Constellation at Eb/No = {ebno} dB")
    
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Baseline Constellation Plot")
    plt.grid(True)
    plt.legend()
    plt.savefig("baseline_constellation_plot.png")
    plt.show()

##################################################
# Loss function plot 
##################################################
def plot_loss_function(loss_file_path,fig_file_path):
    #read the loss values from the file:
    with open(loss_file_path,"rb") as f:
        loss_values = pickle.load(f)
        #Plot the loss curve
    plt.figure()    
    plt.plot(loss_values)
    plt.xlabel("Iterations")
    plt.ylabel("Loss value")
    plt.title("Loss Curve")
    plt.grid("True")
    plt.show()

    #Save the plot:
    plt.savefig(fig_file_path)




##################################################
#calling functions 
##################################################
# Plot the BER and BLER results from "bler_results.pkl"
plot_ber_bler("bler_results.pkl","bler_results_baseline.pkl")

# Plot the constellation before training
plot_constellation("constellation_data.pkl", stage="constellation_before", num_bits_per_symbol=6)

# Plot the constellation after training
plot_constellation("constellation_data.pkl", stage="constellation_after", num_bits_per_symbol=6)

# Plot baseline constellation
plot_constellation_baseline("constellation_baseline.pkl")

#plot loss function:

plot_loss_function("loss_values.pkl","loss_values_plot.pgn")