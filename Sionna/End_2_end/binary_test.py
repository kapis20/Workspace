import tensorflow as tf
from tensorflow.keras import Model
from sionna.utils import BinarySource



binary_source = BinarySource()
uncoded_bits = binary_source([1, 10]) 

print(uncoded_bits)