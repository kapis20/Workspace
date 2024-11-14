
import tensorflow as tf
import numpy 
###############################################
#Padding function
###############################################
def PaddingFunction(tensor,target_length):
    # Calculate the padding amount as a tensor
    padding_amount = tf.maximum(0, target_length - tf.shape(tensor)[1])

    # Create the padding configuration using the padding amount tensor
    paddings = [[0, 0], [0, padding_amount]]

    # Apply padding
    padded_tensor = tf.pad(tensor, paddings, mode="CONSTANT")
    padded_tensor = tf.ensure_shape(padded_tensor, [None, target_length])  # Ensures the second dimension is target_length
    
    return padded_tensor