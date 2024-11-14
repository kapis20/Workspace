
import tensorflow as tf
import numpy 
###############################################
#Padding function
###############################################
def PaddingFunction(tensor,target_length):
    #Calculate the padding ammout:
    Padding_amount = target_length -tf.shape(tensor)[1]
    Padding_amount = int(Padding_amount.numpy())
    padding = tf.constant([[0,0]],[0,Padding_amount])
    padded_tensor = tf.pad(tensor,padding,"CONSTANT")
    
    return padded_tensor