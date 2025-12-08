from polyphase import TwoChannelFilterBank, H0, H1, analysis_filter, analysis_filter_block, synthesis_filter, F0_1, F1_1, F1_2, F0_2, GAIN
from plot_sp import plot_magnitude_response, plot_spec
from signal_gen import generate_input, generate_test_input, rmse
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

# -------------------- Utils -----------------------
def test_at_various_levels(X_n, test):
    delay = 30
    if test == 1:
        # 1 Level Test
        Y_n_top, Y_n_bot = analysis_filter_block(H0, H1, X_n)
        x_n_rec = synthesis_filter(F0_1, F1_1, Y_n_top, Y_n_bot)
    elif test == 2:
        # 2 Level Test
        top1, bot1 = analysis_filter_block(H0, H1, X_n)

        top_t1, bot_t1 = analysis_filter_block(H0, H1, top1)
        top_b1, bot_b1 = analysis_filter_block(H0, H1, bot1)

        top_1_rec = synthesis_filter(F0_1, F1_1, top_t1, bot_t1)
        bot_1_rec = synthesis_filter(F0_1, F1_1, top_b1, bot_b1)

        # Plot top1_rec vs top1
        top1_delayed = np.concatenate((np.zeros(delay), top1[:-delay]))
        plt.plot(top1_delayed.real, label='Original Top1', alpha=0.7)
        plt.plot(top_1_rec.real, label='Reconstructed Top1', alpha=0.7)
        plt.title('Original vs Reconstructed Top1 (Real Part)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Subplots Spectrograms of top1 and top_1_rec
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        pcm1 = plot_spec(ax1, top1_delayed.real, "Original Top1 Spectrogram")
        pcm2 = plot_spec(ax2, top_1_rec.real, "Reconstructed Top1 Spectrogram")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(pcm1, cax=cbar_ax, label='Power (dB)')
        plt.show()

        x_n_rec = synthesis_filter(F0_1, F1_1, top_1_rec, bot1)

    return x_n_rec

# --- 3. THE TEST FUNCTION ---
def run_test(input_sig, title):
    # Analysis
    y_top, y_bot = analysis_filter_block(H0, H1, input_sig)
    
    # Synthesis
    recon_sig = synthesis_filter(F0_1, F1_1, y_top, y_bot)
    
    # Calculate Delay
    delay = 30
    
    # Compare
    # Extract valid region (ignoring start/end transient artifacts)
    valid_len = len(input_sig) - delay
    in_crop = input_sig[:valid_len]
    out_crop = recon_sig[delay:delay+valid_len]
    
    # Error Calculation
    error = in_crop - out_crop
    mse = np.mean(error**2)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(input_sig, 'k', label='Input', alpha=0.5, linewidth=2)
    plt.plot(recon_sig, 'r--', label='Recon (Delayed)')
    plt.title(f"{title} - Visual Check")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(error, 'b')
    plt.title(f"Reconstruction Error (MSE: {mse:.2e})")
    plt.ylim([-0.1, 0.1]) # Zoom in on error
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # --- 4. PLOT SPECTROGRAMS ---
    def plot_spec(ax, x, title):
        # Compute Spectrogram
        f, t_spec, Sxx = signal.spectrogram(x, 40000, nperseg=128, noverlap=120)
        
        # Plot in dB
        pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno', vmin=-80)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        return pcm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    # Plot Input
    plot_spec(ax1, input_sig, "Input Spectrogram (Clean Chirp)")

    # Plot Output
    delay = 30
    valid_recon = recon_sig[delay:] 
    pcm = plot_spec(ax2, valid_recon, "Reconstructed Spectrogram")

    # Add Colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label='Power (dB)')

    plt.show()


# --- Main Execution ---

# 1. Generate a Test Signal
fs = 1024
t = np.linspace(0, 1, fs)
# A mix of low freq (10Hz) and high freq (300Hz)



# 2. Initialize Filter Bank
fb = TwoChannelFilterBank(num_taps=32)

sum = 0
for _ in range(1000):
    # Generate the input signal
    X_n_ap, bin_rep_h = generate_input(1, 1024)
    X_n_h, bin_rep_h = generate_input(2, 1024)
    original_signal = X_n_h

    # 3. Perform Analysis
    low, v3 = fb.analysis(original_signal)
    low_me, v3_me = analysis_filter_block(H0, H1, original_signal)
    
    llow, v2 = fb.analysis(low)  # Second level analysis on high band
    llow_me, v2_me = analysis_filter_block(H0, H1, low)  # Second level analysis on high band

    v0, v1 = fb.analysis(llow)  # Third level analysis on hh band
    v0_me, v1_me = analysis_filter_block(H0, H1, llow)

    # 4. Perform Synthesis
    reconstructed_signal = fb.synthesis(v0, v1)
    reconstructed_signal_me = synthesis_filter(F0_2, F1_2, v0_me, v1_me)

    reconstructed_signal = fb.synthesis(reconstructed_signal, v2)
    reconstructed_signal_me = synthesis_filter(F0_2, F1_2, reconstructed_signal_me, v2_me)

    reconstructed_signal = fb.synthesis(reconstructed_signal, v3)
    reconstructed_signal_me = synthesis_filter(F0_2, F1_2, reconstructed_signal_me, v3_me)

    # 5. Compensate for Filter Delay (Group Delay)
    # For FIR filters of length N, the total delay of Analysis + Synthesis is approx N-1 samples
    delay = 210 + 3
    reconstructed_shifted = reconstructed_signal_me

    # normalized signals for better visualization
    norm_orig = original_signal[:len(reconstructed_shifted[delay:])]
    norm_recon =  reconstructed_shifted[delay:] 

    # calculate RMSE
    sum += rmse(norm_orig, norm_recon)

rmse_value = sum / 1000
print(f"Average RMSE over 1000 runs: {rmse_value:.6f}")
# --- Plotting ---
plt.figure(figsize=(12, 10))

# Plot 1: Original vs Reconstructed
plt.subplot(3, 1, 1)
plt.title("Original vs Reconstructed Signal")
plt.plot(np.linspace(0, len(reconstructed_shifted[delay:]) - 1, len(reconstructed_shifted[delay:])), norm_orig, 'b', label='Original', alpha=0.6, linewidth=2)
# We trim the plot to ignore the edge artifacts
plt.plot(np.linspace(0, len(reconstructed_shifted[delay:]) - 1, len(reconstructed_shifted[delay:])), norm_recon, 'r--', label='Reconstructed (Delay Corrected)')
plt.legend()
plt.grid(True)
plt.show()

  
# Plot 2: Reconstruction Error
plt.subplot(3, 1, 2)
error_signal = original_signal[:len(reconstructed_shifted[delay:])] - reconstructed_shifted[delay:]
plt.title("Reconstruction Error Signal")
plt.plot(np.linspace(0, len(error_signal) - 1, len(error_signal)), error_signal, 'k', label='Error', alpha=0.8)
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Filter Frequency Response
plt.subplot(3, 1, 3)
plt.title("Filter Frequency Responses")
w, h0_resp = signal.freqz(fb.h0)
w, h1_resp = signal.freqz(fb.h1)
plt.plot(w / np.pi, 20 * np.log10(abs(h0_resp)), 'b', label='H0 (Low Pass)')
plt.plot(w / np.pi, 20 * np.log10(abs(h1_resp)), 'r', label='H1 (High Pass)')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Normalized Frequency (x pi rad/sample)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()