import numpy as np

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

