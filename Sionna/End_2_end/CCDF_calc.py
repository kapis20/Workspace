import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
class CCDFCalculator:
    def __init__(self):
        pass
    
    def calculate_instantaneous_power(self, signal):
        """
        Calculate the instantaneous power of the signal.
        Args:
            signal: Tensor of shape [Batch, num_of_samples], complex-valued.

        Returns:
            Tensor of instantaneous power [Batch, num_of_samples].
        """
        return tf.abs(signal) ** 2

    def calculate_average_power(self, instantaneous_power):
        """
        Calculate the average power across the batch.
        Args:
            instantaneous_power: Tensor of shape [Batch, num_of_samples].

        Returns:
            Tensor of average power of shape [Batch, 1].
        """
        return tf.reduce_mean(instantaneous_power, axis=1, keepdims=True)

    def normalize_power(self, instantaneous_power, average_power):
        """
        Normalize the instantaneous power by the average power.
        Args:
            instantaneous_power: Tensor of shape [Batch, num_of_samples].
            average_power: Tensor of shape [Batch, 1].

        Returns:
            Tensor of normalized power of shape [Batch, num_of_samples].
        """
        return instantaneous_power / average_power

    def calculate_ccdf(self, normalized_power, thresholds):
        """
        Calculate the CCDF for the normalized power.
        Args:
            normalized_power: Tensor of shape [Batch, num_of_samples].
            thresholds: List or tensor of thresholds for CCDF calculation.

        Returns:
            Tensor of shape [len(thresholds)], CCDF values for each threshold.
        """
        thresholds = tf.convert_to_tensor(thresholds, dtype=tf.float32)
        batch_size, num_samples = tf.shape(normalized_power)[0], tf.shape(normalized_power)[1]

        # Reshape for broadcasting: [Batch, num_samples, 1] - [1, 1, len(thresholds)]
        normalized_power_expanded = tf.expand_dims(normalized_power, axis=-1)
        thresholds_expanded = tf.reshape(thresholds, [1, 1, -1])

        # Calculate CCDF for each threshold
        exceed_count = tf.reduce_sum(
            tf.cast(normalized_power_expanded > thresholds_expanded, tf.float32),
            axis=1
        )
        ccdf = tf.reduce_mean(exceed_count / tf.cast(num_samples, tf.float32), axis=0)

        return ccdf

    def compute_ccdf(self, signal, thresholds):
        """
        Compute the CCDF for a given signal and thresholds.
        Args:
            signal: Tensor of shape [Batch, num_of_samples], complex-valued.
            thresholds: List or tensor of thresholds for CCDF calculation.

        Returns:
            Tensor of shape [len(thresholds)], CCDF values for each threshold.
        """
        instantaneous_power = self.calculate_instantaneous_power(signal)
        average_power = self.calculate_average_power(instantaneous_power)
        normalized_power = self.normalize_power(instantaneous_power, average_power)
        return self.calculate_ccdf(normalized_power, thresholds)


# ##################################################################
# # Tests 
# ##################################################################



# # Create an instance of the CCDFCalculator (assumes the class code is already defined)
# ccdf_calculator = CCDFCalculator()

# # Generate a sample signal [Batch, num_of_samples]
# batch_size = 10
# num_of_samples = 1000
# signal = tf.complex(
#     tf.random.normal([batch_size, num_of_samples]),
#     tf.random.normal([batch_size, num_of_samples])
# )

# # Define thresholds for CCDF calculation
# thresholds_db = np.linspace(0, 10, 100)  # Thresholds in dB
# thresholds_linear = 10 ** (thresholds_db / 10)  # Convert dB to linear scale

# # Compute CCDF
# ccdf = ccdf_calculator.compute_ccdf(signal, thresholds_linear)

# # Plot the CCDF graph
# plt.figure(figsize=(8, 6))
# plt.plot(thresholds_db, ccdf.numpy(), label='CCDF Curve')

# # Add labels, grid, and title
# plt.xlabel('Normalized Power (dB)', fontsize=12)
# plt.ylabel('CCDF (Probability)', fontsize=12)
# plt.title('CCDF vs Normalized Power', fontsize=14)
# plt.yscale('log')  # Log scale for CCDF (probability)
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.legend()
# plt.show()


##############################################################
## Graphs 
##############################################################



##################################################
# Signal files
##################################################
#NN model:
# File to save the signals
signal_file = "x_rrcf_signals_NN_conv_no_imp.pkl"
#signal_file_noisy = "x_rrcf_Rapp.pkl"
# p = 1
signal_file_noisy1="x_rrcf_signals_RAPP_p_1NN_conv.pkl"
#p = 2 
signal_file_noisy2="x_rrcf_signals_RAPP_p_2NN_conv.pkl"
#p = 3 
signal_file_noisy3="x_rrcf_signals_RAPP_p_3NN_conv.pkl"

#baseline model:
# File to save the signals
signal_file_baseline = "x_rrcf_signals_baseline_no_imp.pkl"
#signal_file_noisy = "x_rrcf_Rapp.pkl"
# p = 1
signal_file_baseline_noisy1="x_rrcf_signals_RAPP_p_1_baseline.pkl"
#p = 2 
signal_file_baseline_noisy2="x_rrcf_signals_RAPP_p_2_baseline.pkl"
#p = 3 
signal_file_baseline_noisy3="x_rrcf_signals_RAPP_p_3_baseline.pkl"

################################################
# Loading signals 
################################################
#NN model 

# Load signals from the file
with open(signal_file, "rb") as f:
    NNloaded_signals = pickle.load(f)


with open(signal_file_noisy1, "rb") as f:
    NNloaded_signals_noisy_p1 = pickle.load(f)

with open(signal_file_noisy2, "rb") as f:
    NNloaded_signals_noisy_p2 = pickle.load(f)


with open(signal_file_noisy3, "rb") as f:
    NNloaded_signals_noisy_p3 = pickle.load(f)


#Baseline: 

with open(signal_file_baseline, "rb") as f:
    Baseline_loaded_signals = pickle.load(f)

with open(signal_file_baseline_noisy1, "rb") as f:
    Baseline_noisy_signals_p1 = pickle.load(f)

with open(signal_file_baseline_noisy2, "rb") as f:
    Baseline_noisy_signals_p2 = pickle.load(f)

with open(signal_file_baseline_noisy3, "rb") as f:
    Baseline_noisy_signals_p3 = pickle.load(f)
###################################################
# Calculate CCDF
###################################################
# Define thresholds for CCDF calculation
#create an array of 100 equally spaced values between 0 and 10
thresholds_db = np.linspace(0, 10, 100)  # Define the range of normalized power thresholds in dB
thresholds_linear = 10 ** (thresholds_db / 10)  # Convert dB thresholds to linear scale

# Compute CCDF for a loaded signal
ccdf_calculator = CCDFCalculator()

# Define signal labels and signal sets for iteration
signal_labels = [
    "E2E no impairment", "E2E p=1", "E2E p=2", "E2E p=3",
    "BL no impairment", "BL p=1", "BL p=2", "BL p=3"
]

signal_sets = [
    NNloaded_signals, NNloaded_signals_noisy_p1, NNloaded_signals_noisy_p2, NNloaded_signals_noisy_p3,
    Baseline_loaded_signals, Baseline_noisy_signals_p1, Baseline_noisy_signals_p2, Baseline_noisy_signals_p3
]

# Plot CCDF for each signal
plt.figure(figsize=(10, 6))
for signals, label in zip(signal_sets, signal_labels):
    # Convert the NumPy signal to a TensorFlow tensor
    x_rrcf_tensor = tf.convert_to_tensor(signals[9])  # Use the same signal index (9) for consistency
    # Compute CCDF for the current signal
    ccdf = ccdf_calculator.compute_ccdf(x_rrcf_tensor, thresholds_linear)
    plt.plot(thresholds_db, ccdf.numpy(), label=label)





# Customize the plot
plt.xlabel(r"Normalized power :$\frac{p(t)}{\bar{p}}$ [dB]", fontsize=12)  # LaTeX formatted x-axis
plt.ylabel(r"CCDF ($P(X > x)$)", fontsize=12)  # LaTeX formatted y-axis
plt.title("CCDF of Normalized Power", fontsize=14)
plt.yscale("log")  # Logarithmic scale for the CCDF
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.xlim(1, 8)
plt.ylim(0.5*1e-5)  # Set y-axis limits to a reasonable range for CCDF
plt.savefig("CCDF_Plot.png", dpi=300)  # Save the plot with an appropriate file extension
plt.show()
