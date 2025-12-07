from polyphase import H0, H1, analysis_filter, analysis_filter_block, synthesis_filter, F0_1, F1_1, F1_2, F0_2
from plot_sp import plot_magnitude_response, plot_spec
from signal_gen import generate_input, generate_test_input, mse
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

