import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle


def Pout_Pin_Power(inputSig, outputSig):
   
    #Normalized
    # inputPower = tf.reduce_mean(tf.abs(inputSig)**2, axis=0, keepdims=True)
    # outputPower = tf.reduce_mean(tf.abs(outputSig)**2, axis=0, keepdims=True)
    # inputPower = (tf.reduce_mean(tf.abs(inputSig), axis=0))**2
    # outputPower =(tf.reduce_mean(tf.abs(outputSig), axis=0))**2

    # inputPower = (tf.abs(tf.reduce_mean(inputSig, axis =0)))**2
    # outputPower =(tf.abs(tf.reduce_mean(outputSig, axis=0)))**2
    inputPower = tf.reduce_mean(tf.abs(inputSig)**2, axis=0)  # Mean power across time/sample axis
    outputPower = tf.reduce_mean(tf.abs(outputSig)**2, axis=0)
    #Convert to dB
    # inputPower = tf.math.log(inputPower) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    # outputPower = tf.math.log(outputPower) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    # inputPower = np.mean(np.abs(inputSig)**2, axis=0)  # Average over signal across corresponding batch signal (columns)
    # outputPower = np.mean(np.abs(outputSig)**2, axis=0)  # Average over signal dimension

   

    return inputPower, outputPower


def Pout_Pin_PowerSingleBatch(inputSig, outputSig):
    inputPower = tf.abs(inputSig)
    inputPower = 20* tf.math.log(inputPower) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) #20 * log 10 
    inputPower = inputPower +30 #dBm 
    outputPower = tf.abs(outputSig)
    outputPower = 20* tf.math.log(outputPower) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) #20 * log 10 
    outputPower = outputPower +30 #dBm 
    # inputPower = tf.abs(inputSig)**2  # Mean power across time/sample axis
    # outputPower = tf.abs(outputSig)**2
   
    # inputPower = tf.square(tf.abs(inputSig))
    # inputPower = 20* tf.math.log(inputPower) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) #20 * log 10 
    # inputPower = inputPower +30 #dBm 

    # outputPower = tf.square(tf.abs(outputSig))
    # outputPower = 20* tf.math.log(outputPower) / tf.math.log(tf.constant(10.0, dtype=tf.float32)) #20 * log 10 
    # outputPower = outputPower +30 #dBm 
    return inputPower, outputPower

#baseline model:
# File to save the signals
#signal_file_noisy = "x_rrcf_Rapp.pkl"
# p = 1
signal_file_baseline_inputP1="x_rrcf_signals_baseline_NEW_inputP=1.pkl"
signal_file_baseline_outputP1 = "x_rrcf_BL_NEW_Rapp_outputP=1.pkl"
#p = 2 
signal_file_baseline_inputP2="x_rrcf_signals_baseline_NEW_inputP=2.pkl"
signal_file_baseline_outputP2 = "x_rrcf_BL_NEW_Rapp_outputP=2.pkl"

#p = 3 
signal_file_baseline_inputP3="x_rrcf_signals_baseline_NEW_inputP=3.pkl"
signal_file_baseline_outputP3 = "x_rrcf_BL_NEW_Rapp_outputP=3.pkl"
#No rapp
signal_file_baseline_new="x_rrcf_signals_baseline_NEW_input.pkl"
signal_file_baseline_output="x_rrcf_BL_NEW_Rapp_output.pkl"


## NN 

#No rapp
signal_file_NN_input="x_rrcf_signals_trained_NN_conv_noRapp.pkl"
signal_file_NN_output="x_rrcf_RappNN_ideal.pkl"

# p = 1
signal_file_NN_inputP1="x_rrcf_signals_trained_NN_conv_P=1(input).pkl"
signal_file_NN_outputP1 = "x_rrcf_RappNN_P=1.pkl"

# p = 2
signal_file_NN_inputP2="x_rrcf_signals_trained_NN_conv_P=2(input).pkl"
signal_file_NN_outputP2 = "x_rrcf_RappNN_P=2.pkl"

# p = 3
signal_file_NN_inputP3="x_rrcf_signals_trained_NN_conv_P=3(input).pkl"
signal_file_NN_outputP3 = "x_rrcf_RappNN_P=3.pkl"


##
signal_file_baseline_inputV15_scaled="x_rrcf_signals_baseline_scaled_inputVs_1_5.pkl"
signal_file_baseline_outputV15_scaled = "x_rrcf_BL_scaled_Rapp_outputVs_1_5.pkl"

signal_file_baseline_inputV1_25_scaled="x_rrcf_signals_baseline_scaled_inputVs_1_25.pkl"
signal_file_baseline_outputV1_25_scaled = "x_rrcf_BL_scaled_Rapp_outputVs_1_25.pkl"

signal_file_baseline_inputV1_scaled="x_rrcf_signals_baseline_scaled_inputVs_1.pkl"
signal_file_baseline_outputV1_scaled = "x_rrcf_BL_scaled_Rapp_outputVs_1.pkl"

signal_file_baseline_input_scaled="x_rrcf_signals_baseline_scaled_input.pkl"
signal_file_baseline_output_scaled = "x_rrcf_BL_scaled_output.pkl"


#########################################################
# Scaled signals BL
#########################################################
signal_file_baseline_input_scaled1="x_rrcf_signals_baseline_scaled_input_V_1.pkl"
signal_file_baseline_output_scaled1 = "x_rrcf_BL_scaled_output_V_1.pkl"

signal_file_baseline_input_scaled3="x_rrcf_signals_baseline_scaled_input_V_3.pkl"
signal_file_baseline_output_scaled3 = "x_rrcf_BL_scaled_output_V_3.pkl"

signal_file_baseline_input_scaled5="x_rrcf_signals_baseline_scaled_input_V_5.pkl"
signal_file_baseline_output_scaled5 = "x_rrcf_BL_scaled_output_V_5.pkl"

#########################################################
# Scaled NN
#########################################################
signal_file_NN_input_scaled1="x_rrcf_signals_trained_NN_conv_scaled_V_1(input).pkl"
signal_file_NN_output_scaled1 = "x_rrcf_RappNN_scaled_V_1.pkl"

signal_file_NN_input_scaled3="x_rrcf_signals_trained_NN_conv_scaled_V_3(input).pkl"
signal_file_NN_output_scaled3 = "x_rrcf_RappNN_scaled_V_3.pkl"

signal_file_NN_input_scaled5="x_rrcf_signals_trained_NN_conv_scaled_V_5(input).pkl"
signal_file_NN_output_scaled5 = "x_rrcf_RappNN_scaled_V_5.pkl"
########################################################
## Scaled BL
########################################################

with open(signal_file_baseline_input_scaled1, "rb") as f:
    Baseline_input_signal_scaled1 = pickle.load(f)

with open(signal_file_baseline_output_scaled1, "rb") as f:
    Baseline_output_signal_scaled1 = pickle.load(f)

with open(signal_file_baseline_input_scaled3, "rb") as f:
    Baseline_input_signal_scaled3 = pickle.load(f)

with open(signal_file_baseline_output_scaled3, "rb") as f:
    Baseline_output_signal_scaled3 = pickle.load(f)

with open(signal_file_baseline_input_scaled5, "rb") as f:
    Baseline_input_signal_scaled5 = pickle.load(f)

with open(signal_file_baseline_output_scaled5, "rb") as f:
    Baseline_output_signal_scaled5 = pickle.load(f)

#########################################################
## Scaled NN 
#########################################################
with open(signal_file_NN_input_scaled1, "rb") as f:
    NN_input_signal_scaled1 = pickle.load(f)

with open(signal_file_NN_output_scaled1, "rb") as f:
    NN_output_signal_scaled1 = pickle.load(f)

with open(signal_file_NN_input_scaled3, "rb") as f:
    NN_input_signal_scaled3 = pickle.load(f)

with open(signal_file_NN_output_scaled3, "rb") as f:
    NN_output_signal_scaled3 = pickle.load(f)

with open(signal_file_NN_input_scaled5, "rb") as f:
    NN_input_signal_scaled5 = pickle.load(f)

with open(signal_file_NN_output_scaled5, "rb") as f:
    NN_output_signal_scaled5 = pickle.load(f)

##########################################################
## Others 
########################################################
with open(signal_file_baseline_input_scaled, "rb") as f:
    Baseline_input_signal_scaled = pickle.load(f)

with open(signal_file_baseline_output_scaled, "rb") as f:
    Baseline_output_signal_scaled = pickle.load(f)


with open(signal_file_baseline_inputV15_scaled, "rb") as f:
    Baseline_input_signalsV15_scaled = pickle.load(f)

with open(signal_file_baseline_outputV15_scaled, "rb") as f:
    Baseline_output_signals_V15_scaled = pickle.load(f)

with open(signal_file_baseline_inputV1_25_scaled, "rb") as f:
    Baseline_input_signalsV1_25_scaled = pickle.load(f)

with open(signal_file_baseline_outputV1_25_scaled, "rb") as f:
    Baseline_output_signals_V1_25_scaled = pickle.load(f)

with open(signal_file_baseline_inputV1_scaled, "rb") as f:
    Baseline_input_signalsV1_scaled = pickle.load(f)

with open(signal_file_baseline_outputV1_scaled, "rb") as f:
    Baseline_output_signals_V1_scaled = pickle.load(f)

#Baseline: 

with open(signal_file_baseline_inputP1, "rb") as f:
    Baseline_input_signalsP1 = pickle.load(f)

with open(signal_file_baseline_outputP1, "rb") as f:
    Baseline_output_signals_p1 = pickle.load(f)

    
with open(signal_file_baseline_inputP2, "rb") as f:
    Baseline_input_signalsP2 = pickle.load(f)

with open(signal_file_baseline_outputP2, "rb") as f:
    Baseline_output_signals_p2 = pickle.load(f)

with open(signal_file_baseline_inputP3, "rb") as f:
    Baseline_input_signalsP3 = pickle.load(f)

with open(signal_file_baseline_outputP3, "rb") as f:
    Baseline_output_signals_p3 = pickle.load(f)

with open(signal_file_baseline_new, "rb") as f:
    Baseline_Input = pickle.load(f)

with open(signal_file_baseline_output, "rb") as f:
    Baseline_Output = pickle.load(f)


## NN
with open(signal_file_NN_input, "rb") as f:
    NN_Input = pickle.load(f)

with open(signal_file_NN_output, "rb") as f:
   NN_Output = pickle.load(f)


with open(signal_file_NN_inputP1, "rb") as f:
    NN_InputP1 = pickle.load(f)

with open(signal_file_NN_outputP1, "rb") as f:
   NN_OutputP1 = pickle.load(f)

with open(signal_file_NN_inputP2, "rb") as f:
    NN_InputP2 = pickle.load(f)

with open(signal_file_NN_outputP2, "rb") as f:
   NN_OutputP2 = pickle.load(f)

with open(signal_file_NN_inputP3, "rb") as f:
    NN_InputP3 = pickle.load(f)

with open(signal_file_NN_outputP3, "rb") as f:
   NN_OutputP3= pickle.load(f)



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

       # Normalize transmit power to 1 Watt per batch
        # power_per_batch = tf.reduce_mean(tf.abs(x_rrcf)**2, axis=1, keepdims=True)
        # scaling_factor = tf.sqrt(1.0 / power_per_batch)
        # scaling_factor=tf.cast(scaling_factor,tf.complex64)
        # x_rrcf = x_rrcf * scaling_factor
# power_per_batch_In = tf.reduce_mean(tf.abs(Baseline_Input[9])**2, axis=1, keepdims=True)
# power_per_batch_Out = tf.reduce_mean(tf.abs(Baseline_Output[9])**2, axis=1, keepdims=True)
# scaling_factor_In = tf.sqrt(1.0 / power_per_batch_In)
# scaling_factor_In=tf.cast(scaling_factor_In,tf.complex64)

# scaling_factor_Out = tf.sqrt(1.0 / power_per_batch_Out)
# scaling_factor_Out=tf.cast(scaling_factor_Out,tf.complex64)


# inputP , outputP = Pout_Pin_Power(Baseline_Input[9]*scaling_factor_In,Baseline_Output[9]*scaling_factor_Out)
# inputP , outputP = Pout_Pin_Power(Baseline_Input[9],Baseline_Output[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="BL no RAPP")
# inputP , outputP = Pout_Pin_Power(Baseline_input_signalsP1[9],Baseline_output_signals_p1[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, P=1")
# inputP , outputP = Pout_Pin_Power(Baseline_input_signalsP2[9],Baseline_output_signals_p2[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, P=2")
# inputP , outputP = Pout_Pin_Power(Baseline_input_signalsP3[9],Baseline_output_signals_p3[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, P=3")
# inputP , outputP = Pout_Pin_Power(NN_Input[9],NN_Output[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="E2E no RAPP")

# inputP , outputP = Pout_Pin_Power(NN_InputP1[9],NN_OutputP1[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP,P=1")

# inputP , outputP = Pout_Pin_Power(NN_InputP2[9],NN_OutputP2[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP,P=2")

# inputP , outputP = Pout_Pin_Power(NN_InputP3[9],NN_OutputP3[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP,P=3")
#print("shape of signal in is"):



##############################################
# Scaled
##############################################

##############################################
# Input vs output magnitude 
##############################################
#tf.print("shape of input is: ",tf.shape(Baseline_input_signal_scaled[9]))
#############################################
# Baseline
#############################################
signalIn1 = Baseline_input_signal_scaled1[9]
signalOut1 =Baseline_output_signal_scaled1[9]

signalIn3 = Baseline_input_signal_scaled3[9]
signalOut3 =Baseline_output_signal_scaled3[9]

signalIn5 = Baseline_input_signal_scaled5[9]
signalOut5 =Baseline_output_signal_scaled5[9]

signalIn1 = signalIn1[0,:]
signalOut1 = signalOut1[0,:]

signalIn3 = signalIn3[0,:]
signalOut3 = signalOut3[0,:]

signalIn5 = signalIn5[0,:]
signalOut5 = signalOut5[0,:]

#############################################
# NN
#############################################
signalIn1NN = NN_input_signal_scaled1[9]
signalOut1NN =NN_output_signal_scaled1[9]

signalIn3NN = NN_input_signal_scaled3[9]
signalOut3NN =NN_output_signal_scaled3[9]

signalIn5NN = NN_input_signal_scaled5[9]
signalOut5NN =NN_output_signal_scaled5[9]

signalIn1NN = signalIn1NN[0,:]
signalOut1NN = signalOut1NN[0,:]

signalIn3NN = signalIn3NN[0,:]
signalOut3NN = signalOut3NN[0,:]

signalIn5NN = signalIn5NN[0,:]
signalOut5NN = signalOut5NN[0,:]
# Powerin = tf.abs(signalIn)**2  
# Powerout = tf.abs(signalOut)**2

# RMSin = tf.sqrt(Powerin)
# RMSOut = tf.sqrt(Powerout)

# Normalizedin = signalIn/RMSin
# Normalizedout = signalOut/RMSOut
# tf.print("shape of input normalized is: ",tf.shape(Normalizedin))


#average across columns (0)
# Powerin = tf.reduce_mean(tf.abs(Baseline_input_signal_scaled[9])**2, axis=0)  # Mean power across time/sample axis
# Powerout = tf.reduce_mean(tf.abs(Baseline_output_signal_scaled[9])**2, axis=0)

# RMSin = tf.sqrt(Powerin)
# RMSOut = tf.sqrt(Powerout)

# Normalizedin = Baseline_input_signal_scaled[9]/RMSin
# Normalizedout = Baseline_output_signal_scaled[9]/RMSOut
# tf.print("shape of input normalized is: ",tf.shape(Normalizedin))

# magnitudes = tf.abs(signalIn1)
# magnitudes_out = tf.abs(signalOut1)
# magnitudes3 = tf.abs(signalIn3)
# magnitudes_out3 = tf.abs(signalOut3)
magnitudes5 = tf.abs(signalIn5)
magnitudes_out5 = tf.abs(signalOut5)
# magnitudes = tf.reduce_mean(tf.abs(Baseline_input_signal_scaled[9]),axis = 0)
# magnitudes_out = tf.reduce_mean(tf.abs(Baseline_output_signal_scaled[9]),axis = 0)
# magnitudes = tf.abs(tf.reduce_mean(Baseline_input_signal_scaled[9], axis =0))
# magnitudes_out = tf.abs(tf.reduce_mean(Baseline_output_signal_scaled[9], axis = 0))
#tf.print("shape of magnitudes is: ",tf.shape(magnitudes))
# Plot the magnitude
real_parts5 = tf.math.real(magnitudes5)
real_parts_out5 = tf.math.real(magnitudes_out5)
plt.figure(figsize=(10, 6))
# plt.plot(magnitudes.numpy(), label='Input')
# plt.plot(magnitudes_out.numpy(), label='Output V=1.0')
# #plt.plot(magnitudes.numpy(), label='Input V=3.0')
# plt.plot(magnitudes_out3.numpy(), label='Output V=3.0')
# plt.plot(magnitudes5.numpy(), label='Input V=5.0')
# plt.plot(magnitudes_out5.numpy(), label='Output V=5.0')
plt.plot(real_parts5.numpy(), label='Input V=5.0')
plt.plot(real_parts_out5.numpy(), label='Output V=5.0')
plt.title('Magnitude of Complex Signal (normalized)')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()
# inputP , outputP = Pout_Pin_Power(Normalizedin,Normalizedout)
# #inputP , outputP = Pout_Pin_Power(Baseline_input_signal_scaled[9],Baseline_output_signal_scaled[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 1.0,p=1")

# inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn1,signalOut1)
# plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 1.0,p=1")
# inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn3,signalOut3)
# plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 3.0,p=1")
inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn5,signalOut5)
#plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 5.0,p=1")
#plt.plot(inputP)  #outputP, alpha=0.5, label="BL RAPP, Vsat = 5.0,p=1")
plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP, Vsat = 5.0,p=1")
plt.xlabel("Input Power (dBm) ")
plt.ylabel("Output Power (dBm)")
plt.title("Output Power vs Input Power (scaled)")
plt.legend()
plt.grid()
plt.show()

inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn5,signalOut5)
plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 5.0,p=1")
inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn3,signalOut3)
plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 3.0,p=1")
inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn1,signalOut1)
plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 1.0,p=1")
# plt.plot(inputP,  inputP, alpha=0.5, label="TX filter")
plt.xlabel("Input Power")
plt.ylabel("Output Power")
plt.title("Output Power vs Input Power")
plt.legend()
plt.grid()
plt.show()

inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn1NN,signalOut1NN)
plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP, Vsat = 1.0,p=1")
inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn3NN,signalOut3NN)
plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP, Vsat = 3.0,p=1")
inputP, outputP = Pout_Pin_PowerSingleBatch(signalIn5NN,signalOut5NN)
plt.plot(inputP,  outputP, alpha=0.5, label="E2E RAPP, Vsat = 5.0,p=1")
plt.xlabel("Input Power ")
plt.ylabel("Output Power")
plt.title("Output Power vs Input Power")
plt.legend()
plt.grid()
plt.show()

inputP , outputP = Pout_Pin_Power(Baseline_input_signalsV15_scaled[9],Baseline_output_signals_V15_scaled[9])
plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 1.5,p=100")
inputP , outputP = Pout_Pin_Power(Baseline_input_signalsV1_25_scaled[9],Baseline_output_signals_V1_25_scaled[9])
plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 1.25,p=100")
inputP , outputP = Pout_Pin_Power(Baseline_input_signalsV1_scaled[9],Baseline_output_signals_V1_scaled[9])
plt.plot(inputP,  outputP, alpha=0.5, label="BL RAPP, Vsat = 1,p=100")

# #print("Sahpe is",Baseline_Input[9].shape)
# plt.plot(10 * np.log10(inputP),  10 * np.log10(outputP), alpha=0.5, label="RAPP P=1")
# inputP , outputP = Pout_Pin_Power(Baseline_loaded_signals[9],Baseline_noisy_signals_p3[9])
# plt.plot(10 * np.log10(inputP),  10 * np.log10(outputP), alpha=0.5, label="RAPP P=3")
# inputP , outputP = Pout_Pin_Power(Baseline_loaded_signals[9],Baseline_noisy_signals_p1[9])

# inputP , outputP = Pout_Pin_Power(Baseline_loaded_signals[9],Baseline_noisy_signals_p3[9])
# plt.plot(inputP,  outputP, alpha=0.5, label="RAPP P=3")



plt.xlabel("Input Power")
plt.ylabel("Output Power")
plt.title("Output Power vs Input Power")
plt.legend()
plt.grid()
plt.show()