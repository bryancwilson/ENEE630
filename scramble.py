import numpy as np
import matplotlib.pyplot as plt
from signal_gen import generate_input
 
def generate_pn_sequences():
    
def generate_pn_sequence(pn_type: int, vector_size: int):
    
    # Initial State of the Register (Key)
    state = 0x3FFFFFF
        
    # Initialize Output
    output = []
    
    # 26 bit Mask
    MASK_26 = 0x3FFFFFF
    
    for i in range(vector_size):
        
        # Get >> nth data
        b25 = (state >> 25) & 1
        b3  = (state >> 3) & 1
        b2  = (state >> 2) & 1
        b1  = (state >> 1) & 1
        b0  = (state >> 0) & 1
        
        if pn_type == 1:
            pn_bit = b25 ^ b3 ^ b0
        elif pn_type == 2:
            pn_bit = b25 ^ b3 ^ b2 ^ b1 ^ b0
            
        output.append(pn_bit)
        
        # Shift Left and Keep to 26 bits
        state = ((state << 1) & MASK_26) | pn_bit
        
    return np.array(output)

def scrambler(input):
    pass

def descrambler(input):
    pass