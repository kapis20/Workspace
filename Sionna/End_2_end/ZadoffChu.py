import numpy as np


### needed for tests ### : 
from scipy.fft import fft
from math import gcd

class ZadoffChuSequence:
    def __init__(self, Nzc, u, q = 0):
           
        """
            Initialize the Zadoff-Chu sequence generator.

            Parameters:
            - Nzc: Length of the ZC sequence (must be a positive integer).
            - u: Root index (integer coprime to Nzc).
            - q: Cyclic shift parameter (default is 0).
        """
            
        self.Nzc = Nzc
        self.u = u
        self.q = q

         
    def generate_zadoff_chu(self):
        """
        Generate a Zadoff-Chu (ZC) sequence.

          

        Returns:
        - ZC sequence as a NumPy array of complex numbers.
        """
            # Ensure root index is valid
        if np.gcd(self.u, self.Nzc) != 1:
            raise ValueError("Root index u must be coprime to the sequence length Nzc.")
        
        # Calculate cf
        cf = self.Nzc % 2 #modulo Nzc / 2 and cf is the remainder 
        
        # Generate the ZC sequence
        n = np.arange(self.Nzc)  # Index range [0, 1, ..., Nzc-1]
        
        zc_sequence = np.exp(-1j * np.pi * self.u * (n + cf + 2 * self.q) * n / self.Nzc)


        return zc_sequence

           
# Example usage
if __name__ == "__main__":
    # Parameters
    Nzc_120GHz = 5  # Sequence length for 120 GHz
    Nzc_220GHz = 8  # Sequence length for 220 GHz
    u = 1           # Root index (coprime to Nzc)
    q = 0           # Cyclic shift parameter (default)

    # Generate sequences
    zc_120GHz = ZadoffChuSequence(Nzc_120GHz, u, q)
    zc_220GHz = ZadoffChuSequence(Nzc_220GHz, u, q)
    
    sequence_120GHz = zc_120GHz.generate_zadoff_chu()
    sequence_220GHz = zc_220GHz.generate_zadoff_chu()
    

    # Print results
    print("Zadoff-Chu Sequence (120 GHz):", sequence_120GHz)
    print("Zadoff-Chu Sequence (220 GHz):", sequence_220GHz)

    print("Shape of 120 is:", sequence_120GHz.shape)
    print("Shape of 220 is:", sequence_220GHz.shape)

    ############################################
    ## Tests 
    ############################################

    #### Periodicity test #####
    # allclose - Returns True if two arrays are element-wise equal within a tolerance.
    # np.roll - shifts elemeents by 5 or 8 so NZC 

    periodicity_120 = np.allclose(sequence_120GHz, np.roll(sequence_120GHz, 5))  # Shift by Nzc
    periodicity_220 = np.allclose(sequence_220GHz, np.roll(sequence_220GHz, 8))  # Shift by Nzc

    print("Periodic with period Nzc 120:", periodicity_120)
    print("Periodic with period Nzc 220:", periodicity_220)

    # Expected results: Periodic with period Nzc: True


    ### Discrete Fourier Transform (DFT) Property ###

    def multiplicative_inverse(u, Nzc):
        for i in range(Nzc):
            if (u * i) % Nzc == 1:
                return i
        return None

    #DFT of ZC sequences 
    X_u_120 = fft(sequence_120GHz)  # DFT of ZC sequence
    X_u_220 = fft(sequence_220GHz)  # DFT of ZC sequence

    # Generate conjugated and time-scaled version
    u_inverse_120 = multiplicative_inverse(u, Nzc_120GHz)
    u_inverse_220 = multiplicative_inverse(u, Nzc_220GHz)

    reconstructed_Xu_120 = np.conj(sequence_120GHz[u_inverse_120 * np.arange(Nzc_120GHz) % Nzc_120GHz]) * X_u_120[0]
    reconstructed_Xu_220 = np.conj(sequence_220GHz[u_inverse_220 * np.arange(Nzc_220GHz) % Nzc_220GHz]) * X_u_220[0]

    print("DFT property 120 holds:", np.allclose(X_u_120, reconstructed_Xu_120))
    print("DFT property 220 holds:", np.allclose(X_u_220, reconstructed_Xu_220))

    #Expected output: DFT property holds: True

    ### Autocorellation Propert: ###
    #Compute the autocorrelation of the ZC sequence and verify that only the peak at 𝜏 = 0 τ=0 is non-zero.

    autocorrelation_120 = np.correlate(sequence_120GHz, sequence_120GHz, mode='full')  # Full autocorrelation
    center = len(autocorrelation_120) // 2  # Center corresponds to τ = 0
    is_autocorrelation_correct_120 = np.allclose(autocorrelation_120[:center] + autocorrelation_120[center+1:], 0)
    autocorrelation_220 = np.correlate(sequence_220GHz, sequence_220GHz, mode='full')  # Full autocorrelation
    center = len(autocorrelation_220) // 2  # Center corresponds to τ = 0
    is_autocorrelation_correct_220 = np.allclose(autocorrelation_220[:center] + autocorrelation_220[center+1:], 0)
    print("Autocorrelation property 120 holds:", is_autocorrelation_correct_120)
    print("Autocorrelation property 220 holds:", is_autocorrelation_correct_220)

    #Expected output: Autocorrelation property holds: True
    
    ### Cross - Correlation property ###:
    u2 = 3
    zc_u2_120 = ZadoffChuSequence(Nzc_120GHz, u2).generate_zadoff_chu()
    zc_u2_220 = ZadoffChuSequence(Nzc_220GHz, u2).generate_zadoff_chu()

    # cross_correlation = np.abs(np.sum(sequence_120GHz * np.conj(zc_u2_120))) / np.sqrt(Nzc_120GHz)
    # expected_value = 1 / np.sqrt(Nzc_120GHz)
    # print("Cross-correlation 120 value:", cross_correlation)
    # # isclose - returns a boolean array where two arrays are element-wise equal within a tolerance
    # print("Cross-correlation 120 matches expected:", np.isclose(cross_correlation, expected_value))

    # cross_correlation = np.abs(np.sum(sequence_220GHz * np.conj(zc_u2_220))) / np.sqrt(Nzc_220GHz)
    # expected_value = 1 / np.sqrt(Nzc_220GHz)
    # print("Cross-correlation 220 value:", cross_correlation)
    # # isclose - returns a boolean array where two arrays are element-wise equal within a tolerance
    # print("Cross-correlation 220 matches expected:", np.isclose(cross_correlation, expected_value))

    # cross_correlation_120 = np.abs(np.sum(sequence_120GHz * np.conj(zc_u2_120))) / np.sqrt(Nzc_120GHz)
    # expected_value_120 = 1 / np.sqrt(Nzc_120GHz)
    # print("Cross-correlation 120 value:", cross_correlation_120)
    # print("Cross-correlation 120 matches expected:", np.isclose(cross_correlation_120, expected_value_120, atol=1e-6))

    # cross_correlation_220 = np.abs(np.sum(sequence_220GHz * np.conj(zc_u2_220))) / np.sqrt(Nzc_220GHz)
    # expected_value_220 = 1 / np.sqrt(Nzc_220GHz)
    # print("Cross-correlation 220 value:", cross_correlation_220)
    # print("Cross-correlation 220 matches expected:", np.isclose(cross_correlation_220, expected_value_220, atol=1e-6))






