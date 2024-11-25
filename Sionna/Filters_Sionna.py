import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.config.seed = 42



import matplotlib
matplotlib.use('TkAgg')  # You can replace 'TkAgg' with another supported backend

import matplotlib.pyplot as plt
import numpy as np

from sionna.utils import QAMSource
from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr



beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbold
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

rrcf.show("impulse")
rrcf.show("magnitude", "db") # Logarithmic scale
rrcf.show("magnitude", "lin") # Linear scale

print("Filter length:", rrcf.length)

# Configure QAM source
num_bits_per_symbol = 4 # The modulation order of the QAM constellation, i.e., 16QAM
qam = QAMSource(num_bits_per_symbol) # Layer to generate batches of QAM symbols

# Generate batch of QAM symbol sequences
batch_size = 128
num_symbols = 1000
x = qam([batch_size, num_symbols])
print("Shape of x", x.shape)

# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)

# Upsample the QAM symbol sequence
x_us = us(x)
print("Shape of x_us", x_us.shape)

# Inspect the first few elements of one row of x_us
plt.stem(np.abs(x_us)[0,:20]);
plt.xlabel(r"Sample index $i$")
plt.ylabel(r"|$x_{us}[i]$|");

# Filter the upsampled sequence
x_rrcf = rrcf(x_us)

# Apply the matched filter
x_mf = rrcf(x_rrcf)

# Instantiate a downsampling layer
ds = Downsampling(samples_per_symbol, rrcf.length-1, num_symbols)

# Recover the transmitted symbol sequence
x_hat = ds(x_mf)
print("Shape of filtered upsampled is:", x_rrcf.shape)
print("Shape of mathced filter is", x_mf.shape)
print("Shape of downsampled signal is", x_hat.shape)

# Visualize the different signals
plt.figure(figsize=(12, 8))
plt.plot(np.real(x_us[0]), "x")
plt.plot(np.real(x_rrcf[0, rrcf.length//2:]))
plt.plot(np.real(x_mf[0, rrcf.length-1:]));
plt.xlim(0,100)
plt.legend([r"Oversampled sequence of QAM symbols $x_{us}$",
            r"Transmitted sequence after pulse shaping $x_{rrcf}$",
            r"Received sequence after matched filtering $x_{mf}$"]);

plt.show()