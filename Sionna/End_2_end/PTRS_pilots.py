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
        return tf.concat(blocks_with_ptrs, axis=1)

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
