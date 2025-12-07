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

def mse(X_n, Xk_n):
    error = X_n - Xk_n
    mse_value = np.mean(np.abs(error)**2)
    return mse_value

