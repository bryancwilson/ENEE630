import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input, generate_test_input, rmse
from polyphase import filter_bank, filter_bank_2, calculate_MSE, H0, H1, F0_1, F1_1, F0_2, F1_2
from plot_sp import filter_bank_plots, plot_signal, plot_spec, plot_T

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
# plot_T(H0, H1, F0_2, F1_2)

# ----------------------------------- MSE Filterbank -----------------------------------
calculate_MSE(1, add_noise=False, snr=0, scrambling=False, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=1000)
calculate_MSE(2, add_noise=False, snr=0, scrambling=False, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=1000)
calculate_MSE(1, add_noise=False, snr=0, scrambling=False, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=1000)
calculate_MSE(2, add_noise=False, snr=0, scrambling=False, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=1000)


f_offsets = [0, -2*78.125, 2*78.125, 25] 
snrs = range(21)

# Initialize storage for results
# Each list contains 4 sub-lists (one for each offset)
mses_f1 = [[], [], [], []] # Input 1, Filter Set 1
mses_f2 = [[], [], [], []] # Input 2, Filter Set 1
mses_f3 = [[], [], [], []] # Input 1, Filter Set 2
mses_f4 = [[], [], [], []] # Input 2, Filter Set 2

print("Starting Simulation...")

for i, offset in enumerate(f_offsets):
    print(f"  Simulating Offset: {offset} Hz")
    for snr in snrs:
        # 1. Filter Set 1 (using H0, H1, F0_1, F1_1)
        mses_f1[i].append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=False, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))
        mses_f2[i].append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=False, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))

        # 2. Filter Set 2 (using H0_2, H1_2, F0_2, F1_2)
        mses_f3[i].append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=False, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))
        mses_f4[i].append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=False, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))

# --- Plotting ---
plt.figure(figsize=(16, 12))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.semilogy(range(21), mses_f1[i], 'o-', label='Input Type 1, Filter Set 1')
    plt.semilogy(range(21), mses_f2[i], 's-', label='Input Type 2, Filter Set 1')
    plt.semilogy(range(21), mses_f3[i], 'o-', label='Input Type 1, Filter Set 2') # Fixed Label
    plt.semilogy(range(21), mses_f4[i], 's-', label='Input Type 2, Filter Set 2') # Fixed Label
    plt.title('Frequency Offset: {:.2f} Hz'.format(f_offsets[i]))
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

plt.tight_layout()
plt.show()



# # -------------------------------------- Apply Scrambling ----------------------------------------
calculate_MSE(1, add_noise=False, snr=0, scrambling=True, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=1000)
calculate_MSE(2, add_noise=False, snr=0, scrambling=True, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=1000)
calculate_MSE(1, add_noise=False, snr=0, scrambling=True, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=1000)
calculate_MSE(2, add_noise=False, snr=0, scrambling=True, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=1000)


f_offsets = [0, -2*78.125, 2*78.125, 25] 
snrs = range(21)

# Initialize storage for results
# Each list contains 4 sub-lists (one for each offset)
mses_f1 = [[], [], [], []] # Input 1, Filter Set 1
mses_f2 = [[], [], [], []] # Input 2, Filter Set 1
mses_f3 = [[], [], [], []] # Input 1, Filter Set 2
mses_f4 = [[], [], [], []] # Input 2, Filter Set 2

print("Starting Simulation...")

for i, offset in enumerate(f_offsets):
    print(f"  Simulating Offset: {offset} Hz")
    for snr in snrs:
        # 1. Filter Set 1 (using H0, H1, F0_1, F1_1)
        mses_f1[i].append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=True, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))
        mses_f2[i].append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=True, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))

        # 2. Filter Set 2 (using H0_2, H1_2, F0_2, F1_2)
        mses_f3[i].append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=True, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))
        mses_f4[i].append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=True, freq_shift=offset, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))

# --- Plotting ---
plt.figure(figsize=(16, 12))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.semilogy(range(21), mses_f1[i], 'o-', label='Input Type 1, Filter Set 1')
    plt.semilogy(range(21), mses_f2[i], 's-', label='Input Type 2, Filter Set 1')
    plt.semilogy(range(21), mses_f3[i], 'o-', label='Input Type 1, Filter Set 2') # Fixed Label
    plt.semilogy(range(21), mses_f4[i], 's-', label='Input Type 2, Filter Set 2') # Fixed Label
    plt.title('Frequency Offset: {:.2f} Hz'.format(f_offsets[i]))
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Squared Error (MSE) With Scrambling')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

plt.tight_layout()
plt.show()
