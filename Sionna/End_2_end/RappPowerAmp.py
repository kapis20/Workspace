import tensorflow as tf


class RappPowerAmplifier:
    """Model of a power amplifier according to Rapp's model for TensorFlow tensors."""

    def __init__(self, saturation_amplitude: float, smoothness_factor: float = 1.0, voltage_gain: float =19.0):
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
        self.voltage_gain = voltage_gain
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
        amplitude = tf.abs(input_signal)
        gain = tf.pow(1 + tf.pow((self.voltage_gain*amplitude) / self.saturation_amplitude, 2 * p), 1 / (2 * p))
        gain = self.voltage_gain*amplitude / gain
        # Convert gain to complex64 to match input_signal
        gain = tf.cast(gain, dtype=input_signal.dtype)

        return input_signal * gain

