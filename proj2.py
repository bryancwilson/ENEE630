import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input
from polyphase import analysis_filter, analysis_filter_block, synthesis_filter, H0, H1, F0_1, F1_1, F0_2, F1_2
from plot_sp import filter_bank_plots, plot_signal

# Generate the input signal
X_n, bin_rep = generate_input(1, 4096)
plot_signal(X_n)
# ----------------------------------- Apply Polyphase Filter Bank -----------------------------------
# Analysis Filter Bank
Y_n = analysis_filter(H0, H1, X_n)
# First Level of Decomposition
Y_n_level_1_top, Y_n_level_1_bot = analysis_filter_block(H0, H1, X_n)
# Second Level of Decomposition
Y_n_level_2_top, Y_n_level_2_bot = analysis_filter_block(H0, H1, Y_n_level_1_top)
# Third Level of Decomposition
Y_n_level_3_top, Y_n_level_3_bot = analysis_filter_block(H0, H1, Y_n_level_2_top)

# Flatten the lists (since your code appended list of arrays/floats)
v3 = np.array(Y_n_level_1_bot).flatten()
v2 = np.array(Y_n_level_2_bot).flatten()
v1 = np.array(Y_n_level_3_bot).flatten()
v0 = np.array(Y_n_level_3_top).flatten()

# Synthesis Filter Bank Reconstruction
x_n_rec_top = synthesis_filter(F0_1, F1_1, v0, v1)
x_n_rec_mid = synthesis_filter(F0_1, F1_1, x_n_rec_top, v2)
x_n_rec = synthesis_filter(F0_1, F1_1, x_n_rec_mid, v3)

# ----------------------------------- Results and Plots -----------------------------------
mse_value = np.mean(np.abs(X_n - x_n_rec)**2)
print(f"MSE between original and reconstructed signal: {mse_value}")    
# Plot Filter Bank Responses
filter_bank_plots(H0, H1)