import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input
from plot_sp import filter_bank_plots

# High-Pass and Low-Pass Filter Coefficients
LPF = [-1, 0, 3, 0, -8, 0, 21, 0, -45, 0, 91, 0, -191, 0, 643, 1024, 643, 0, -191, 0, 91, 0, -45, 0, 21, 0, -8, 0, 3, 0, -1]
HPF = [1, 0, -3, 0, 8, 0, -21, 0, 45, 0, -91, 0, 191, 0, -643, 1024, -643, 0, 191, 0, -91, 0, 45, 0, -21, 0, 8, 0, -3, 0, 1]

def polyphase_filter(LPF, HPF, X_n, section=1):
    M = 2  # Number of phases
    L = len(LPF) // M  # Length of each polyphase component

    # Create polyphase components
    E0_lpf = np.array([LPF[i] for i in range(0, len(LPF), M)])
    E1_lpf = np.array([LPF[i] for i in range(1, len(LPF), M)])
    E0_hpf = np.array([HPF[i] for i in range(0, len(HPF), M)])
    E1_hpf = np.array([HPF[i] for i in range(1, len(HPF), M)])

    # Initialize output signal
    Y_n_top = []
    Y_n_bot = []

    # Polyphase filtering
    for n in range(0, len(X_n), M):
        # Get input samples
        x0 = X_n[n] if n < len(X_n) else 0
        x1 = X_n[n + 1] if (n + 1) < len(X_n) else 0

        # Apply polyphase components
        y0_top = np.dot(E0_lpf, [x0] + [X_n[n - i] if (n - i) >= 0 else 0 for i in range(1, len(E0_lpf))])
        y1_top = np.dot(E1_lpf, [x1] + [X_n[n - i + 1] if (n - i + 1) >= 0 else 0 for i in range(1, len(E1_lpf))])
        y0_bot = np.dot(E0_hpf, [x0] + [X_n[n - i] if (n - i) >= 0 else 0 for i in range(1, len(E0_hpf))])
        y1_bot = np.dot(E1_hpf, [x1] + [X_n[n - i + 1] if (n - i + 1) >= 0 else 0 for i in range(1, len(E1_hpf))])
        
        Y_n_top.append(y0_top + y1_top) # Output of top branch
        Y_n_bot.append(y0_bot + y1_bot) # output of bottom branch

        # recursion (decompose low branch further)
    if section == 3:
        return ['v1', [Y_n_bot]], ['v0', [Y_n_top]]
    else:
        high_tree = polyphase_filter(LPF, HPF, Y_n_top, section=section+1)
        return high_tree, ['v'+str(3 - section + 1), [Y_n_bot]]

X_n, bin_rep = generate_input(2, 1024)
Y_n = polyphase_filter(LPF, HPF, X_n)
y_n = [Y_n[0][0][0], Y_n[0][0][1], Y_n[0][1], Y_n[1]] 

filter_bank_plots(HPF, LPF)