import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from RappPowerAmp import RappPowerAmplifier

def Pout_Pin_Power(inputSig, outputSig):
    """
    Compute the average power of input and output signals.
    Args:
        inputSig (tf.Tensor): Input signal tensor.
        outputSig (tf.Tensor): Output signal tensor.
    Returns:
        tuple: Average input power and average output power.
    """
    # Compute input and output power (normalized over the entire signal)
    inputPower = np.mean(np.abs(inputSig.numpy())**2)  # Average over entire signal
    outputPower = np.mean(np.abs(outputSig.numpy())**2)  # Average over entire signal
    return inputPower, outputPower



# # Generate the tensor for x^2 function
# x_values = np.linspace(0, 2, 500)  # 500 points between 0 and 2
# y_values = x_values ** 2
# input_signal = tf.constant(y_values, dtype=tf.float32)



# Generate a complex signal
np.random.seed(42)  # For reproducibility
real_part = np.linspace(0, 1, 500)  # Real part linearly spaced
imag_part = np.sin(np.linspace(0, 2 * np.pi, 500))  # Imaginary part as sine wave
complex_signal = real_part + 1j * imag_part  # Create complex numbers
input_signal = tf.constant(complex_signal, dtype=tf.complex64)






# Smoothness factors to evaluate
p_factors = [1, 2, 5, 10]


# Calculate P_in and P_out for each smoothness factor
input_powers = []
output_powers = []

# Create a dictionary to store input vs output signals
input_vs_output = {}

##############################
# Complex numbers 
#############################

input_vs_output = {}

for p in p_factors:
    # Initialize the RappPowerAmplifier
    rapp_amp = RappPowerAmplifier(saturation_amplitude=1.0, smoothness_factor=p)
    # Process the input signal through the power amplifier
    amplified_signal = rapp_amp(input_signal)
    # Compute input and output power
    input_power, output_power = Pout_Pin_Power(input_signal, amplified_signal)
    input_powers.append(input_power)
    output_powers.append(output_power)
    # Store input vs output data for plotting
    input_vs_output[p] = amplified_signal.numpy()

# Create subplots for the two graphs
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: P_out vs. P_in
axes[0].plot(input_powers, output_powers, marker="o")
axes[0].set_xlabel("Input Power (P_in)")
axes[0].set_ylabel("Output Power (P_out)")
axes[0].set_title("P_out vs. P_in for Different Smoothness Factors (p)")
axes[0].grid(True)

# Annotate points with corresponding smoothness factors
for i, p in enumerate(p_factors):
    axes[0].annotate(f"p={p}", (input_powers[i], output_powers[i]), textcoords="offset points", xytext=(5, -10))

# Plot 2: Input vs. Output
for p, amplified_signal in input_vs_output.items():
    axes[1].plot(np.real(input_signal.numpy()), np.real(amplified_signal), label=f"Amplified Signal (p={p})")
axes[1].plot(np.real(input_signal.numpy()), np.real(input_signal.numpy()), label="Original Signal", linestyle="--", color="black")
axes[1].set_xlabel("Input Signal (Real Part)")
axes[1].set_ylabel("Output Signal (Real Part)")
axes[1].set_title("Input vs. Output (Real Part) for Different Smoothness Factors (p)")
axes[1].legend()
axes[1].grid(True)

# Display the plots
plt.tight_layout()
plt.show()
###############################
# Real numbers
##############################
# for p in p_factors:
#     # Initialize the RappPowerAmplifier
#     rapp_amp = RappPowerAmplifier(saturation_amplitude=1.0, smoothness_factor=p)
#     # Process the input signal through the power amplifier
#     amplified_signal = rapp_amp(input_signal)
#     # Compute input and output power
#     input_power, output_power = Pout_Pin_Power(input_signal, amplified_signal)
#     input_powers.append(input_power)
#     output_powers.append(output_power)
#     # Store input vs output data for plotting
#     input_vs_output[p] = amplified_signal.numpy()

# # Create subplots for the two graphs
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# # Plot 1: P_out vs. P_in
# axes[0].plot(input_powers, output_powers, marker="o")
# axes[0].set_xlabel("Input Power (P_in)")
# axes[0].set_ylabel("Output Power (P_out)")
# axes[0].set_title("P_out vs. P_in for Different Smoothness Factors (p)")
# axes[0].grid(True)

# # Annotate points with corresponding smoothness factors
# for i, p in enumerate(p_factors):
#     axes[0].annotate(f"p={p}", (input_powers[i], output_powers[i]), textcoords="offset points", xytext=(5, -10))

# # Plot 2: Input vs. Output
# for p, amplified_signal in input_vs_output.items():
#     axes[1].plot(x_values, amplified_signal, label=f"Amplified Signal (p={p})")
# axes[1].plot(x_values, y_values, label="Original Signal (y = x^2)", linestyle="--", color="black")
# axes[1].set_xlabel("Input Signal (x)")
# axes[1].set_ylabel("Output Signal (y)")
# axes[1].set_title("Input vs. Output for Different Smoothness Factors (p)")
# axes[1].legend()
# axes[1].grid(True)

# # Display the plots
# plt.tight_layout()
# plt.show()