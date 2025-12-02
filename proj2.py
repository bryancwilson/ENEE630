import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input, generate_test_input, mse
from polyphase import analysis_filter, analysis_filter_block, synthesis_filter, H0, H1, F0_1, F1_1, F0_2, F1_2
from plot_sp import filter_bank_plots, plot_signal, plot_spec
from scramble import generate_pn_sequence

# Generate the input signal
X_n_ap, bin_rep_h = generate_input(1, 4096)
X_n_h, bin_rep_h = generate_input(2, 1000)
# plot_signal(X_n_h,
#             stochastic=False,
#             title="Input Signal (Band Pass)")
X_n = X_n_ap
_, X_n = generate_test_input()

# Analyze the Filters
# filter_bank_plots(H0, H1)
# ----------------------------------- Apply Polyphase Filter Bank -----------------------------------
# ----- Analysis Filter Bank -----
Y_n = analysis_filter(H0, H1, X_n)
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

# -------------------------------------- Apply Scrambling ----------------------------------------
generate_pn_sequence(pn_type=1, vector_size=1025)
# ----------------------------------- Results and Plots -----------------------------------

# plot_signal(X_n,
#             stochastic=False,
#             title="Original Signal")
# plot_signal(x_n_rec,    
#             stochastic=False,
#             title="Reconstructed Signal")
delay = 90
x_n_rec_corrected = x_n_rec[delay - 1:] # Adjust for delay introduced by filtering
X_n_corrected = X_n[:len(X_n) - delay + 1]

# Plot Signals In Time
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(X_n_corrected.real, label='Original Signal (Real)', alpha=0.7)
plt.plot(x_n_rec_corrected.real, label='Reconstructed Signal (Real)', alpha=0.7)
plt.title('Original vs Reconstructed Signal (Real Part)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot Signals In Frequency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
plot_spec(ax1, X_n_corrected.real, "Input Spectrogram (Clean Chirp)")
pcm = plot_spec(ax2, x_n_rec_corrected.real, "Reconstructed Spectrogram")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(pcm, cax=cbar_ax, label='Power (dB)')
plt.show()

mse_value = mse(X_n_corrected, x_n_rec_corrected)
print(f"MSE between original and reconstructed signal: {mse_value}")    
