import tensorflow as tf


class RappPowerAmplifier:
    """Model of a power amplifier according to Rapp's model for TensorFlow tensors."""

    def __init__(self, saturation_amplitude: float, smoothness_factor: float = 1.0):
        """
        Args:
            saturation_amplitude (float): Maximum amplitude before saturation occurs.
            smoothness_factor (float, optional): Smoothness factor of the amplification 
                                                 saturation characteristics. Must be > 0.
        """
        if smoothness_factor <= 0.0:
            raise ValueError("Smoothness factor must be greater than zero.")
        self.saturation_amplitude = saturation_amplitude
        self.smoothness_factor = smoothness_factor
        
    #__call__ makes sure you can call it in other scripts 
    def __call__(self, input_signal: tf.Tensor) -> tf.Tensor:
        """
        Apply the power amplifier model to the input signal using Rapp's model.

        Args:
            input_signal (tf.Tensor): Input tensor of shape (batch, num_samples).

        Returns:
            tf.Tensor: Amplified output signal of the same shape as input.
        """
        p = self.smoothness_factor
        pin = 10.**(-30/10)# unit: W
        amplitude = tf.abs(input_signal)
        gain = tf.pow(1 + tf.pow(amplitude / self.saturation_amplitude, 2 * p), 1 / (2 * p))
        tf.print("gain is ",tf.shape(gain))
        tf.print("gain dtype is ", gain.dtype)
        #gain = gain * tf.cast(tf.sqrt(pin), dtype=gain.dtype)
        # Convert gain to complex64 to match input_signal
        gain = tf.cast(gain, dtype=input_signal.dtype)
        tf.print("gain is ",tf.shape(gain))
        # Print the first 10 values (or fewer, depending on the tensor size)
         # Print the real part of gain
        tf.print("gain (real part, first 10):", tf.math.real(gain)[:, 10:])

        # Print the imaginary part of gain
        tf.print("gain (imaginary part, firat 10):", tf.math.imag(gain)[:, 10:])
        # Print the real part of input
        tf.print("input (real part, first 10):", tf.math.real(input_signal)[:,10:])

        # Print the imaginary part of input
        tf.print("input (imaginary part, first 10):", tf.math.imag(input_signal)[:,10:])

        # tf.print("gain last 10):", gain[: -10:])
        # tf.print("input last 10):", input_signal[: -10:])
        return input_signal / gain

