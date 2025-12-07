import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input, generate_test_input, mse
from polyphase import filter_bank, filter_bank_2, H0, H1, F0_1, F1_1, F0_2, F1_2
from plot_sp import filter_bank_plots, plot_signal, plot_spec, plot_T
from scramble import generate_pn_sequence
from test_bench import test_at_various_levels

# Generate the input signal
X_n_ap, bin_rep_h = generate_input(1, 1024)
X_n_h, bin_rep_h = generate_input(2, 1024)
# plot_signal(X_n_h,
#             stochastic=False,
#             title="Input Signal (Band Pass)")

_, X_n = generate_test_input()
X_n = X_n_h

# Analyze the Filters
# filter_bank_plots(H0, H1)
# plot_T(H0, H1, F0_1, F1_1)

# ----------------------------------- Apply Polyphase Filter Bank -----------------------------------
N = 1000
sum = 0
delay = 30
for _ in range(N):
    # x_hat = filter_bank_2(X_n)
    x_hat = test_at_various_levels(X_n, 1)

    # --- Alignment and MSE Calculation ---
    delay = 30 
    x_n_rec_corrected = x_hat[delay - 1:] 
    X_n_corrected = X_n[:len(X_n) - delay + 1]

    min_len = min(len(X_n_corrected), len(x_n_rec_corrected))
    X_n_corrected = X_n_corrected[:min_len]
    x_n_rec_corrected = x_n_rec_corrected[:min_len]

    mse_value = mse(X_n_corrected, x_n_rec_corrected)

    # --- Plotting ---
    error = X_n_corrected - x_n_rec_corrected

    # --- Plot 1: Original Signal (Top) ---
    plt.subplot(2, 1, 1) # 2 rows, 1 column, index 1
    plt.plot(X_n_corrected.real, label='Original Signal (Real)', color='blue', alpha=0.7)
    plt.plot(x_n_rec_corrected.real, label='Reconstructed Signal (Real)', color='orange', alpha=0.7)   
    plt.title('Original Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # --- Plot 2: Reconstructed Signal (Bottom) ---
    plt.subplot(2, 1, 2) # 2 rows, 1 column, index 2 (The bottom slot)
    plt.plot(x_n_rec_corrected.real, label='Reconstructed Signal (Real)', color='orange', alpha=0.7)
    plt.title('Reconstructed Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout() # Adjusts spacing so titles don't overlap
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(error), label='Magnitude of Reconstruction Error (|X_n - x_hat|)', color='red', linewidth=1)
    plt.title(f'Reconstruction Error Magnitude (MSE = {mse_value:.2e})')
    plt.xlabel('Sample Index (n)')
    plt.ylabel('Error Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.yscale('log')
    plt.show()

    # Rolling Sum
    sum += mse_value

print(f"MSE between original and reconstructed signal: {sum/N}")    
# -------------------------------------- Apply Scrambling ----------------------------------------
# generate_pn_sequence(pn_type=1, vector_size=1025)

# ----------------------------------- Results and Plots -----------------------------------

x_n_rec = test_at_various_levels(X_n, 1)
plot_signal(X_n,
            stochastic=False,
            title="Original Signal")
plot_signal(x_n_rec,    
            stochastic=False,
            title="Reconstructed Signal")
delay = 30
x_n_rec_corrected = x_n_rec[delay - 1:] # Adjust for delay introduced by filtering
X_n_corrected = X_n[:len(X_n) - delay + 1]

# Plot Signals In Time
plt.figure(figsize=(12, 8))

# --- Plot 1: Original Signal (Top) ---
plt.subplot(2, 1, 1) # 2 rows, 1 column, index 1
plt.plot(X_n_corrected.real, label='Original Signal (Real)', color='blue', alpha=0.7)
plt.title('Original Signal')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# --- Plot 2: Reconstructed Signal (Bottom) ---
plt.subplot(2, 1, 2) # 2 rows, 1 column, index 2 (The bottom slot)
plt.plot(x_n_rec_corrected.real, label='Reconstructed Signal (Real)', color='orange', alpha=0.7)
plt.title('Reconstructed Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout() # Adjusts spacing so titles don't overlap
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
