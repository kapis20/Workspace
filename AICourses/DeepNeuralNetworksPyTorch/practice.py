import torch 
import numpy as np 
import pandas as pd

# import matplotlib.pyplot as plt
# %matplotlib inline  


A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)

print("The result is:", A_times_B)