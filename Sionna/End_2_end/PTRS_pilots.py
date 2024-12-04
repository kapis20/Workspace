import tensorflow as tf
import numpy as np

from ZadoffChu import  ZadoffChuSequence

class PTRSPilotInserter:
    def __init__(self, Nzc_PTRS, u_PTRS, Q, batch_size,num_symbols_per_codeword):
        """
        Initialize PTRS Pilot Inserter parameters.
        Args:
            Nzc_PTRS: Length of Zadoff-Chu sequence for PTRS.
            u_PTRS: Root index for PTRS.
            Q: Number of Q blocks.
            batch_size: Batch size of the data.
        """
        self.Nzc_PTRS = Nzc_PTRS
        self.u_PTRS = u_PTRS
        self.Q = Q
        self.batch_size = batch_size
        self.num_symbols_per_codeword = num_symbols_per_codeword

        # Use the ZadoffChuSequence class to generate the PTRS sequence
        self.ptrs_sequence = ZadoffChuSequence(Nzc_PTRS, u_PTRS).generate_zadoff_chu()

    def add_ptrs_to_blocks(self, data):
        """
        Add PTRS pilots to the signal blocks.
        Args:
            data: Input QAM data (tensor of shape [batch_size, num_symbols]).
        Returns:
            A tensor with PTRS pilots added to each block.
        """
        #num_symbols_per_codeword = tf.shape(data)[1]
        data_symbols_per_block = self.num_symbols_per_codeword // self.Q

        # Split data into Q blocks
        data_blocks = tf.split(data, self.Q, axis=1)

        # Expand PTRS sequence for batch size
        ptrs_batch = tf.tile(
            tf.convert_to_tensor(self.ptrs_sequence, dtype=tf.complex64)[tf.newaxis, :],
            [self.batch_size, 1]
        )

        # Add PTRS to each block
        blocks_with_ptrs = [
            tf.concat([ptrs_batch, block], axis=1) for block in data_blocks
        ]
        # Concatenate all blocks with PTRS
        signal_with_ptrs = tf.concat(blocks_with_ptrs, axis=1)
         # Format transmitted PTRS (match dimensions: [batch_size, Q, Nzc_PTRS])
        transmitted_ptrs = tf.tile(
        tf.convert_to_tensor(self.ptrs_sequence, dtype=tf.complex64)[tf.newaxis, tf.newaxis, :],
        [self.batch_size, self.Q, 1]
        )
        
        return signal_with_ptrs, transmitted_ptrs

# # Example usage
# if __name__ == "__main__":
#     # Parameters
#     Nzc_PTRS = 4  # Length of Zadoff-Chu sequence for PTRS
#     u_PTRS = 1    # Root index for PTRS
#     Q = 32        # Number of Q blocks
#     BATCH_SIZE = 8  # Example batch size
#     NUM_SYMBOLS = 1024  # Example number of symbols

#     # Generate example data
#     data = tf.random.uniform((BATCH_SIZE, NUM_SYMBOLS), dtype=tf.complex64)

#     # Initialize PTRS inserter
#     ptrs_inserter = PTRSPilotInserter(Nzc_PTRS, u_PTRS, Q, BATCH_SIZE)

#     # Add PTRS pilots to data
#     data_with_ptrs = ptrs_inserter.add_ptrs_to_blocks(data)

#     # Print results
#     print("Data with PTRS shape:", data_with_ptrs.shape)



class PhaseNoiseCompensator:
    def __init__(self, num_ptrs_per_group):
        """
        Initialize the Phase Noise Compensator.
        
        Args:
            num_ptrs_per_group: Number of PTRS samples per group (Np).
        """
        self.num_ptrs_per_group = num_ptrs_per_group

    def calculate_phase_error(self, received_ptrs, transmitted_ptrs):
        """
        Calculate the average phase error for each PTRS group.
        
        Args:
            received_ptrs: Received PTRS symbols (complex tensor of shape [batch_size, Q, Np]).
            transmitted_ptrs: Transmitted PTRS symbols (complex tensor of shape [batch_size, Q, Np]).
        
        Returns:
            Phase error estimates for each group (tensor of shape [batch_size, Q]).
        """
        # Element-wise conjugate of transmitted PTRS
        conj_transmitted_ptrs = tf.math.conj(transmitted_ptrs)
        
        # Compute the numerator and denominator for phase estimation
        numerator = tf.reduce_sum(received_ptrs * conj_transmitted_ptrs, axis=-1)
        denominator = tf.reduce_sum(tf.abs(transmitted_ptrs) ** 2, axis=-1)
        
        # Cast denominator to tf.complex64 for division
        denominator = tf.cast(denominator, dtype=tf.complex64)
        # Compute the average phase error for each group
        phase_error = tf.math.angle(numerator / denominator*self.num_ptrs_per_group)
        return phase_error

    def interpolate_phase_error(self, phase_error, num_data_symbols):
        """
        Interpolate the phase error estimates over the entire block.
        
        Args:
            phase_error: Phase error estimates for each group (tensor of shape [batch_size, Q]).
            num_data_symbols: Total number of data symbols (integer).
        
        Returns:
            Interpolated phase error (tensor of shape [batch_size, num_data_symbols]).
        """
        batch_size, num_groups = tf.shape(phase_error)[0], tf.shape(phase_error)[1]
        interpolated_phase_error = tf.image.resize(
            phase_error[:, :, tf.newaxis],  # Add channel dimension
            [batch_size, num_data_symbols],  # Target size
            method='bilinear'
        )
        return tf.squeeze(interpolated_phase_error, axis=-1)  # Remove the channel dimension

    def compensate_phase_noise(self, received_signal, interpolated_phase_error):
        """
        Compensate the phase noise in the received signal.
        
        Args:
            received_signal: Received signal (complex tensor of shape [batch_size, num_symbols]).
            interpolated_phase_error: Interpolated phase error (tensor of shape [batch_size, num_symbols]).
        
        Returns:
            Phase-compensated signal (tensor of shape [batch_size, num_symbols]).
        """
        # Cast interpolated phase error to complex64
        interpolated_phase_error = tf.cast(interpolated_phase_error, dtype=tf.complex64)


        phase_correction = tf.exp(-1j * interpolated_phase_error)
        compensated_signal = received_signal * phase_correction
        return compensated_signal
