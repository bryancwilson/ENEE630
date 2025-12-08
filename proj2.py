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

mse_option1_input1 = []
mse_option1_input2 = []
mse_option2_input1 = []
mse_option2_input2 = []
for snr in range(21):
    mse_option1_input1.append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=False, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))
    mse_option1_input2.append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=False, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))
    mse_option2_input1.append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=False, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))
    mse_option2_input2.append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=False, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))

# plot MSE vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(range(21), mse_option1_input1, 'o-', label='Input Type 1, Filter Set 1')
plt.semilogy(range(21), mse_option1_input2, 's-', label='Input Type 2, Filter Set 1')
plt.semilogy(range(21), mse_option2_input1, 'o--', label='Input Type 1, Filter Set 2')
plt.semilogy(range(21), mse_option2_input2, 's--', label='Input Type 2, Filter Set 2')
plt.title('MSE vs SNR for Different Input Types and Filter Sets')
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

mse_option1_input1 = []
mse_option1_input2 = []
mse_option2_input1 = []
mse_option2_input2 = []
for snr in range(21):
    mse_option1_input1.append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=True, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))
    mse_option1_input2.append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=True, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1, N=100))
    mse_option2_input1.append(calculate_MSE(1, add_noise=True, snr=snr, scrambling=True, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))
    mse_option2_input2.append(calculate_MSE(2, add_noise=True, snr=snr, scrambling=True, H0=H0, H1=H1, F0_1=F0_2, F1_1=F1_2, N=100))

# plot MSE vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(range(21), mse_option1_input1, 'o-', label='Input Type 1, Filter Set 1')
plt.semilogy(range(21), mse_option1_input2, 's-', label='Input Type 2, Filter Set 1')
plt.semilogy(range(21), mse_option2_input1, 'o--', label='Input Type 1, Filter Set 2')
plt.semilogy(range(21), mse_option2_input2, 's--', label='Input Type 2, Filter Set 2')
plt.title('MSE vs SNR for Different Input Types and Filter Sets')
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()