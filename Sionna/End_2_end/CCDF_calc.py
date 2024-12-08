import tensorflow as tf

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


##################################################################
# Tests 
##################################################################
import matplotlib.pyplot as plt
import numpy as np


# Create an instance of the CCDFCalculator (assumes the class code is already defined)
ccdf_calculator = CCDFCalculator()

# Generate a sample signal [Batch, num_of_samples]
batch_size = 10
num_of_samples = 1000
signal = tf.complex(
    tf.random.normal([batch_size, num_of_samples]),
    tf.random.normal([batch_size, num_of_samples])
)

# Define thresholds for CCDF calculation
thresholds_db = np.linspace(0, 10, 100)  # Thresholds in dB
thresholds_linear = 10 ** (thresholds_db / 10)  # Convert dB to linear scale

# Compute CCDF
ccdf = ccdf_calculator.compute_ccdf(signal, thresholds_linear)

# Plot the CCDF graph
plt.figure(figsize=(8, 6))
plt.plot(thresholds_db, ccdf.numpy(), label='CCDF Curve')

# Add labels, grid, and title
plt.xlabel('Normalized Power (dB)', fontsize=12)
plt.ylabel('CCDF (Probability)', fontsize=12)
plt.title('CCDF vs Normalized Power', fontsize=14)
plt.yscale('log')  # Log scale for CCDF (probability)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
