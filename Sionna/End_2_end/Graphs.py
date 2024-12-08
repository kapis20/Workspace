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
    ax1.semilogy(ebno_dbs_nn, BER_nn, 'x--', c='C1', label='BER - autoencoder-NN')
    ax1.semilogy(ebno_dbs_baseline, BER_baseline, 'x--', c='C2', label='BER - Baseline')
    ax1.axvline(6.82, color='red', linestyle='--', label="Shannon's Band")
    ax1.set_ylabel("BER")
    ax1.grid(which="both", linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_ylim((1e-5, 1))

    # Plot BLER on the bottom subplot
    ax2.semilogy(ebno_dbs_nn, BLER_nn, 'o-', c='C0', label='BLER - autoencoder-NN')
    ax2.semilogy(ebno_dbs_baseline, BLER_baseline, 'o-', c='C3', label='BLER - Baseline')
    ax2.axvline(6.82, color='red', linestyle='--', label="Shannon's Band")
    ax2.set_xlabel(r"$E_b/N_0$ (dB)")
    ax2.set_ylabel("BLER")
    ax2.grid(which="both", linestyle='--', linewidth=0.5)
    ax2.legend()
    ax2.set_ylim((1e-5, 1))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("ber_bler_combined_plot.png")
    plt.show()

#plot only for NN demapper 
def plot_ber_bler_NN(results_filename):
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

   

        # Extract eb/no values, BLER, and BER specifically for 'autoencoder-NN'
        ebno_dbs = results['ebno_dbs']['autoencoder-NN']
        BLER = results['BLER']['autoencoder-NN']
        BER = results['BER']['autoencoder-NN']

        # Create a figure with two subplots, one for BER and one for BLER
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # Plot BER on the top subplot
        ax1.semilogy(ebno_dbs, BER, 'x--', c='C1', label='BER - autoencoder-NN')
        ax1.axvline(6.8, color='red', linestyle='--', label="Shannon's Band")
        ax1.set_ylabel("BER")
        ax1.grid(which="both", linestyle='--', linewidth=0.5)
        ax1.legend()
        ax1.set_ylim((1e-4, 1.0))

        # Plot BLER on the bottom subplot
        ax2.semilogy(ebno_dbs, BLER, 'o-', c='C0', label='BLER - autoencoder-NN')
        ax2.set_xlabel(r"$E_b/N_0$ (dB)")
        ax2.axvline(6.8, color='red', linestyle='--', label="Shannon's Band")
        ax2.set_ylabel("BLER")
        ax2.grid(which="both", linestyle='--', linewidth=0.5)
        ax2.legend()
        ax2.set_ylim((1e-4, 1.0))

        
      
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig("ber_bler_NN_plot.png")
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
        plt.scatter(real, imag)#, label=f"Constellation at Eb/No = {ebno} dB")
    
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("64 QAM Constellation Plot")
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
     #Save the plot:
    plt.savefig(fig_file_path)
    plt.show()

   

####################################################
# BLER BER baseline 
##################################################
import pickle
import matplotlib.pyplot as plt

def plot_baseline_ber_bler(file_names, papr_limits):
    """
    Plots BER and BLER for multiple baseline results corresponding to different PAPR limits.
    
    Parameters:
        file_names (list of str): List of file paths to the baseline result files.
        papr_limits (list of float): List of corresponding PAPR limits.
    """
    # Create a figure with two subplots, one for BER and one for BLER
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Loop through the result files and plot data
    for file_name, papr in zip(file_names, papr_limits):
        # Load data from the current file
        with open(file_name, 'rb') as f:
            baseline_results = pickle.load(f)

        # Extract baseline data
        ebno_dbs_baseline = baseline_results['ebno_dbs']['baseline']
        BLER_baseline = baseline_results['BLER']['baseline']
        BER_baseline = baseline_results['BER']['baseline']

        # Add BER and BLER to the respective subplots
        label_suffix = f"PAPR={papr}"
        ax1.semilogy(ebno_dbs_baseline, BER_baseline, label=f"BER - Baseline ({label_suffix})")
        ax2.semilogy(ebno_dbs_baseline, BLER_baseline, label=f"BLER - Baseline ({label_suffix})")

    # Format BER subplot
    ax1.axvline(6.82, color='red', linestyle='--', label="Shannon's Band")
    ax1.set_ylabel("BER")
    ax1.grid(which="both", linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_ylim((1e-5, 1))

    # Format BLER subplot
    ax2.axvline(6.82, color='red', linestyle='--', label="Shannon's Band")
    ax2.set_xlabel(r"$E_b/N_0$ (dB)")
    ax2.set_ylabel("BLER")
    ax2.grid(which="both", linestyle='--', linewidth=0.5)
    ax2.legend()
    ax2.set_ylim((1e-5, 1))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("ber_bler_baseline_papr_plot.png")
    plt.show()

def plot_single_baseline_ber_bler(file_name, papr_limit):
    """
    Plots BER and BLER for a single baseline result file with a specified PAPR limit.
    
    Parameters:
        file_name (str): File path to the baseline result file.
        papr_limit (float): The PAPR limit for the result file.
    """
    import matplotlib.pyplot as plt
    import pickle

    # Load data from the file
    with open(file_name, 'rb') as f:
        baseline_results = pickle.load(f)

    # Extract baseline data
    ebno_dbs_baseline = baseline_results['ebno_dbs']['baseline']
    BLER_baseline = baseline_results['BLER']['baseline']
    BER_baseline = baseline_results['BER']['baseline']

    # Create a figure with two subplots, one for BER and one for BLER
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot BER
    ax1.semilogy(ebno_dbs_baseline, BER_baseline, label=f"BER - Baseline (PAPR={papr_limit})")
    ax1.axvline(6.82, color='red', linestyle='--', label="Shannon's Band")
    ax1.set_ylabel("BER")
    ax1.grid(which="both", linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_ylim((1e-5, 1))

    # Plot BLER
    ax2.semilogy(ebno_dbs_baseline, BLER_baseline, label=f"BLER - Baseline (PAPR={papr_limit})")
    ax2.axvline(6.82, color='red', linestyle='--', label="Shannon's Band")
    ax2.set_xlabel(r"$E_b/N_0$ (dB)")
    ax2.set_ylabel("BLER")
    ax2.grid(which="both", linestyle='--', linewidth=0.5)
    ax2.legend()
    ax2.set_ylim((1e-5, 1))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"ber_bler_baseline_papr_{papr_limit}_plot.png")
    plt.show()




##################################################
#calling functions 
##################################################
# Plot the BER and BLER results from "bler_results.pkl"
# plot_ber_bler("bler_results.pkl","bler_results_baseline.pkl")
# #plot_ber_bler_NN("bler_results.pkl")

# # Plot the constellation before training
# plot_constellation("constellation_data.pkl", stage="constellation_before", num_bits_per_symbol=6)

# # Plot the constellation after training
# plot_constellation("constellation_data.pkl", stage="constellation_after", num_bits_per_symbol=6)

# # Plot baseline constellation
# #plot_constellation_baseline("constellation_baseline.pkl")

# #plot loss function:

# plot_loss_function("loss_values.pkl","loss_values_plot.png")

# Example usage
# file_names = [
#     'bler_results_baseline5_5.pkl',
#     'bler_results_baseline6_0.pkl',
#     'bler_results_baseline6_5.pkl'
# ]
# papr_limits = [5.5, 6.0, 6.5]

# plot_baseline_ber_bler(file_names, papr_limits)

#plot_single_baseline_ber_bler('bler_results_baseline.pkl',5.5)

plot_constellation_baseline("constellation_data_QAM64.pkl")