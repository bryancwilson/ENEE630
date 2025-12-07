import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input
from plot_sp import filter_bank_plots
from scipy import signal

# High-Pass and Low-Pass Filter Coefficients
GAIN = 2050
H0 = np.array([-1, 0, 3, 0, -8, 0, 21, 0, -45, 0, 91, 0, -191, 0, 643, 1024, 643, 0, -191, 0, 91, 0, -45, 0, 21, 0, -8, 0, 3, 0, -1])
H1 = np.array([-1, 0, 3, 0, -8, 0, 21, 0, -45, 0, 91, 0, -191, 0, 643, -1024, 643, 0, -191, 0, 91, 0, -45, 0, 21, 0, -8, 0, 3, 0, -1])

num_taps = 30
#H0 = signal.firwin(num_taps, cutoff=0.5, window='hamming')
alternating_signs = np.ones(num_taps)
alternating_signs[1::2] = -1  # Set indices 1, 3, 5... to -1
#H1 = H0 * alternating_signs

# First Set of Synthesis Filters
F0_1 = np.array(H0)
F1_1 = np.array(H1)

# Second Set of Synthesis Filters
F0_2 = np.array(H0)
F1_2 = np.array(H1)

def filter_bank(X_n):
    # ----- Analysis Filter Bank -----
    # First Level of Decomposition
    Y_n_level_1_top, Y_n_level_1_bot = analysis_filter_block(H0, H1, X_n)
    # Second Level of Decomposition
    Y_n_level_2_top, Y_n_level_2_bot = analysis_filter_block(H0, H1, Y_n_level_1_top)
    # Third Level of Decomposition
    Y_n_level_3_top, Y_n_level_3_bot = analysis_filter_block(H0, H1, Y_n_level_2_top)

    # Rename According to Report Notation
    v3 = np.array(Y_n_level_1_bot).flatten()
    v2 = np.array(Y_n_level_2_bot).flatten()
    v1 = np.array(Y_n_level_3_bot).flatten()
    v0 = np.array(Y_n_level_3_top).flatten()

    # ----- Synthesis Filter Bank -----
    # First Stage of Reconstruction
    x_n_rec_top = synthesis_filter(F0_1, F1_1, v0, v1)
    # Second Stage of Reconstruction
    x_n_rec_mid = synthesis_filter(F0_1, F1_1, x_n_rec_top, v2)
    # Final Stage of Reconstruction
    x_n_rec = synthesis_filter(F0_1, F1_1, x_n_rec_mid, v3)

    return x_n_rec

def analysis_filter_block(H0, H1, X_n):
    M = 2  # Number of phases
    L = len(H0) // M  # Length of each polyphase component

    # Create polyphase components
    E0_H0 = np.array([H0[i] for i in range(0, len(H0), M)])
    E1_H0 = np.array([H0[i] for i in range(1, len(H0), M)])
    E0_H1 = np.array([H1[i] for i in range(0, len(H1), M)])
    E1_H1 = np.array([H1[i] for i in range(1, len(H1), M)])

    # Split input into even and odd samples (This is the decimation step)
    x_even = X_n[0::M]
    
    # We pad with 0 if odd length to ensure even/odd arrays match size (This is the split and delay step)
    if len(X_n) % 2 != 0:
        x_odd = np.pad(X_n[1::M], (0, 1), 'constant')
    else:
        x_odd = X_n[1::M]
    
    # (This is the filtering step and adding step)
    # High Pass Branch
    y_bot_even = signal.lfilter(E0_H1, 1, x_even)
    y_bot_odd  = signal.lfilter(E1_H1, 1, x_odd)
    Y_n_bot = y_bot_even + y_bot_odd

    # Low Pass Branch
    y_top_even = signal.lfilter(E0_H0, 1, x_even)
    y_top_odd  = signal.lfilter(E1_H0, 1, x_odd)
    Y_n_top = y_top_even + y_top_odd

    return Y_n_top / (GAIN), Y_n_bot / (GAIN)

def synthesis_filter(F0, F1, Y_n_top, Y_n_bot):
    L = 2  # Number of phases

    # Create polyphase components
    R0_H0 = np.array([F0[i] for i in range(1, len(F0), L)])
    R1_H0 = np.array([F0[i] for i in range(0, len(F0), L)])
    R0_H1 = np.array([F1[i] for i in range(1, len(F1), L)])
    R1_H1 = np.array([F1[i] for i in range(0, len(F1), L)])

    # (This is the filtering step and adding step)
    # Low Pass Branch
    x_recon_top_H0 = signal.lfilter(R0_H0, 1, Y_n_top)
    x_recon_bot_H0 = signal.lfilter(R1_H0, 1, Y_n_top)

    # High Pass Branch
    x_recon_top_H1 = signal.lfilter(R0_H1, 1, Y_n_bot)
    x_recon_bot_H1 = signal.lfilter(R1_H1, 1, Y_n_bot)

    x_recon_even = x_recon_top_H0 + x_recon_top_H1
    x_recon_odd = x_recon_bot_H0 + x_recon_bot_H1

    # Delay and Combine Branches
    x_reconstructed = np.zeros(2 * len(Y_n_top)) # Use complex if inputs are complex
    
    x_reconstructed[0::2] = x_recon_even
    x_reconstructed[1::2] = x_recon_odd

    return x_reconstructed / (GAIN / 2)

def analysis_filter(H0, H1, X_n, section=1):
    M = 2  # Number of phases
    L = len(H0) // M  # Length of each polyphase component

    # Create polyphase components
    E0_H0 = np.array([H0[i] for i in range(0, len(H0), M)])
    E1_H0 = np.array([H0[i] for i in range(1, len(H0), M)])
    E0_H1 = np.array([H1[i] for i in range(0, len(H1), M)])
    E1_H1 = np.array([H1[i] for i in range(1, len(H1), M)])

    # Split input into even and odd samples
    x_even = X_n[0::M]
    
    # We pad with 0 if odd length to ensure even/odd arrays match size
    if len(X_n) % 2 != 0:
        x_odd = np.pad(X_n[1::M], (0, 1), 'constant')
    else:
        x_odd = X_n[1::M]
    
    # Low Pass Branch
    y_low_even = signal.lfilter(E0_H0, 1, x_even)
    y_low_odd  = signal.lfilter(E1_H0, 1, x_odd)
    Y_n_bot = y_low_even + y_low_odd

    # High Pass Branch
    y_high_even = signal.lfilter(E0_H1, 1, x_even)
    y_high_odd  = signal.lfilter(E1_H1, 1, x_odd)
    Y_n_top = y_high_even + y_high_odd

    # recursion (decompose low branch further)
    if section == 3:
        return ['v1', [Y_n_bot]], ['v0', [Y_n_top]]
    else:
        high_tree = analysis_filter(H0, H1, Y_n_top, section=section+1)
        return high_tree, ['v'+str(3 - section + 1), [Y_n_bot]]