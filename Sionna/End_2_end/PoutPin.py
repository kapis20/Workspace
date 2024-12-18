import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle


def Pout_Pin_Power(inputSig, outputSig):
    # inputPower = np.mean(np.abs(inputSig)**2, axis=0)  # Average over signal across corresponding batch signal (columns)
    # outputPower = np.mean(np.abs(outputSig)**2, axis=0)  # Average over signal dimension
    inputPower = np.mean(np.abs(inputSig)**2, axis=0)  # Average over signal across corresponding batch signal (columns)
    outputPower = np.mean(np.abs(outputSig)**2, axis=0)  # Average over signal dimension

    # inputPower_flat = inputPower.flatten()  # Shape: (batchSize * signal,)
    # outputPower_flat = outputPower.flatten()  # Shape: (batchSize * signal,)

    return inputPower, outputPower


#baseline model:
# File to save the signals
signal_file_baseline = "x_rrcf_signals_baseline_no_imp.pkl"
#signal_file_noisy = "x_rrcf_Rapp.pkl"
# p = 1
signal_file_baseline_noisy1="x_rrcf_signals_RAPP_p_1_baseline.pkl"
#p = 2 
signal_file_baseline_output="x_rrcf_BL_NEW_Rapp_output.pkl"
#p = 3 
signal_file_baseline_new="x_rrcf_signals_baseline_NEW_input.pkl"



#Baseline: 

with open(signal_file_baseline, "rb") as f:
    Baseline_loaded_signals = pickle.load(f)

with open(signal_file_baseline_noisy1, "rb") as f:
    Baseline_noisy_signals_p1 = pickle.load(f)

with open(signal_file_baseline_new, "rb") as f:
    Baseline_Input = pickle.load(f)

with open(signal_file_baseline_output, "rb") as f:
    Baseline_Output = pickle.load(f)





#     # Define signal labels and signal sets for iteration
# signal_labels = [
#     #"E2E no impairment", "E2E p=1", "E2E p=2", "E2E p=3",
#     "BL no impairment", "BL p=1", "BL p=2", "BL p=3",
#     #"E2E RAPP, p=1", "E2E RAPP, p=3"
# ]

# signal_sets = [
#     #NNloaded_signals, NNloaded_signals_noisy_p1, NNloaded_signals_noisy_p2, NNloaded_signals_noisy_p3,
#     Baseline_loaded_signals, Baseline_noisy_signals_p1, Baseline_noisy_signals_p2, Baseline_noisy_signals_p3,
#     #NNloaded_signals_noisy_RAPP_p1, NNloaded_signals_noisy_RAPP_p3
# ]


inputP , outputP = Pout_Pin_Power(Baseline_Input[9],Baseline_Output[9])
print("Sahpe is",Baseline_Input[9].shape)
# plt.plot(10 * np.log10(inputP),  10 * np.log10(outputP), alpha=0.5, label="RAPP P=1")
# inputP , outputP = Pout_Pin_Power(Baseline_loaded_signals[9],Baseline_noisy_signals_p3[9])
# plt.plot(10 * np.log10(inputP),  10 * np.log10(outputP), alpha=0.5, label="RAPP P=3")
# inputP , outputP = Pout_Pin_Power(Baseline_loaded_signals[9],Baseline_noisy_signals_p1[9])
plt.plot(inputP,  outputP, alpha=0.5, label="BL no RAPP")
# inputP , outputP = Pout_Pin_Power(Baseline_loaded_signals[9],Baseline_noisy_signals_p3[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="RAPP P=3")



plt.xlabel("Input Power")
plt.ylabel("Output Power")
plt.title("Output Power vs Input Power")
plt.legend()
plt.grid()
plt.show()