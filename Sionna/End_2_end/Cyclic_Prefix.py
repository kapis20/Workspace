import tensorflow as tf
import numpy as np

class CyclicPrefix:
    def __init__(self, cp_ratio, signal_length):
        """
        Initialize the Cyclic Prefix (CP) parameters.
        :param cp_ratio: The ratio of CP length to the block length.
        """
        self.cp_ratio = cp_ratio
        self.signal_length = signal_length

    def add_cp(self, signal):
        """
        Add cyclic prefix to the input signal.
        :param signal: Tensor of shape [batch_size, signal_length].
        :return: Tensor with CP added, shape [batch_size, signal_length + cp_length].
        """
        # Convert self.cp_ratio to a TensorFlow tensor
        cp_ratio = tf.constant(self.cp_ratio, dtype=tf.float32)
         # Calculate CP length using TensorFlow operations
        #signal_length = tf.cast(tf.shape(signal)[-1], tf.float32)
        cp_length = tf.cast(cp_ratio * self.signal_length / (1 + cp_ratio), tf.int32)
         # Extract CP from the end
        cp = signal[:, -cp_length:]  # Shape: [batch_size, cp_length]
        return tf.concat([cp, signal], axis=1)

    def remove_cp(self, signal):
        """
        Remove cyclic prefix from the input signal.
        :param signal: Tensor of shape [batch_size, signal_length + cp_length].
        :return: Tensor without CP, shape [batch_size, signal_length].
        """

        cp_ratio = tf.constant(self.cp_ratio, dtype=tf.float32)
         # Calculate CP length using TensorFlow operations
        #total_length = tf.cast(tf.shape(signal)[-1], tf.float32)  # Cast to float32
        cp_length = tf.cast(cp_ratio * (self.signal_length / (1 + cp_ratio)), tf.int32)  # Compute and cast back to int32

        return signal[:, cp_length:]
