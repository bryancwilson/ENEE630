import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
def generate_input(type: int, vector_size: int):
    # Paramters
    lower_bound = -1*(2**15)
    upper_bound = (2**15 - 1)
    n = np.linspace(0, vector_size - 1, vector_size)

    if type == 1:
        # Generate I and Q Component Separately
        x_i = np.random.randint(lower_bound, upper_bound, size=vector_size, dtype=np.int32)
        x_q = np.random.randint(lower_bound, upper_bound, size=vector_size, dtype=np.int32)
        # x_q = np.zeros(vector_size)  # Q Channel is Zero

        # Combine To Form Complex Signal
        X_n = x_i + 1j * x_q

    elif type == 2:
        # Generate I and Q Component Separately
        f = [0.06, 0.18, 0.38, 0.8]
        a = [1344, 864, 8543, -43]
        x_i = np.zeros(vector_size)
        x_q = np.zeros(vector_size)
        for f_, a_ in zip(f, a):
            c = np.cos(np.pi * f_ * n)
            s = np.sin(np.pi * f_ * n)

            x_i += a_*c
            x_q += a_*s

        # Combine To Form Complex Signal
        X_n = x_i + 1j * x_q

    # Binary Representation
    bin_rep = []
    for val in X_n:
        # Convert Valyes to 16-bit Binary (twos complement)
        binary16_real = format(int(val.real) & 0xFFFF, '016b')
        binary16_imag = format(int(val.imag) & 0xFFFF, '016b')

        # Append Tuple to List
        bin_rep.append((binary16_real, binary16_imag))

    return X_n, bin_rep

def generate_test_input():
    N = 20000 # Number of samples
    duration = 1.0  # seconds
    f = N/duration  # Frequency
    fs = f*2  # Sampling frequency
    t = np.linspace(0, duration, int(fs)) # Time vector

    # Create an impulse signal
    impulse = np.zeros(31)
    impulse[0] = 100  

    # Frequency Sweep (Chirp) Signal
    chirp_sig = signal.chirp(t, f0=0, t1=1, f1=f, method='linear')

    return impulse, chirp_sig

def rmse(x_ref, x_recon):
    """
    Calculates the custom error metric defined in the image.
    Formula: sqrt( sum( |x - x_hat|^2 / |x|^2 ) ) / 1024
    """
    # Ensure inputs are numpy arrays
    x = np.array(x_ref)
    hat_x = np.array(x_recon)
    
    # Safety Check: The formula hardcodes 1024, but your signal might vary.
    # We will use 1024 to strictly follow the image, or len(x) if you prefer genericism.
    N = 1024 
    
    # 1. Numerator: Squared magnitude of the difference (Error Power)
    # Using np.abs() handles complex numbers correctly
    diff_sq = np.abs(x - hat_x)**2
    
    # 2. Denominator: Squared magnitude of the reference (Signal Power)
    ref_sq = np.abs(x)**2
    
    # Avoid division by zero (add small epsilon where ref is 0)
    ref_sq[ref_sq == 0] = 1e-10 
    
    # 3. Element-wise Division (Relative Error)
    relative_error = diff_sq / ref_sq
    
    # 4. Sum, Sqrt, and Scale
    sum_val = np.sum(relative_error)
    result = np.sqrt(sum_val) / N
    
    return result

