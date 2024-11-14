import tensorflow as tf


t = tf.constant([[1, 2, 3], [4, 5, 6]])
print(t)

paddings = tf.constant([[0,0],[0,192]])
#first corresponds to rows dimension and by [0 0 ] we say that we want nothing added before the starting row and after
#secodn corresponds to columns dimension an we say 0 before each sequence but two 0s at the end of each sequence 
print("Paddings is ",paddings)

x =tf.constant(2)

tensor = tf.zeros((2,2), dtype=x.dtype)

# Set the last element to the value of x
# Set the last element to the value of x using tf.tensor_scatter_nd_update
indices = [[1, 1]]  # Position of the last element
updates = [x]       # The value of x
padding2 = tf.tensor_scatter_nd_update(tensor, indices, updates)

print("Paddings 2 is", padding2)
#padding2 = tensor + tf.constant([[0, 0], [0, x]])

t_new = tf.pad(t,paddings,"CONSTANT")

tnew1 = tf.pad(t,padding2,"CONSTANT")

print("new t is",t_new)
print("new1 is ",tnew1)

print("shaoe is ", int(tf.shape(t_new)[1]))


paddings = tf.constant([[0,0],[0,int(tf.shape(t_new)[1])]])

print("new paddings:",paddings)

n = 100 - tf.shape(t_new)[1]

# print("N is",n)

# integer_from_tensor = int(n.numpy())
# print("n.numpy is", integer_from_tensor)

# reshaped_tensor = tf.reshape(n, (1,))
# # Print the reshaped tensor
# print(reshaped_tensor)
# print("New shape:", reshaped_tensor.shape)