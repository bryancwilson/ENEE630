from polyphase import H0, H1, analysis_filter, analysis_filter_block, synthesis_filter, F0_1, F1_1, F1_2, F0_2, GAIN
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

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class TwoChannelFilterBank:
    def __init__(self, num_taps=32):
        """
        Initialize the filter bank with a prototype Low Pass Filter.
        Using a standard QMF design approach.
        """
        # 1. Design Prototype Low Pass Filter (H0)
        # We use an EVEN length (ODD order) filter to satisfy QMF constraints
        self.h0 = H0

        # 2. Design High Pass Filter (H1)
        # H1[n] = (-1)^n * H0[n] (Mirror filter)
        # This shifts the frequency response by pi
        self.h1 = H1
        

        # 3. Design Synthesis Filters (F0, F1)
        # For alias cancellation in QMF:
        # F0 is typically 2 * H0 (to compensate for downsampling energy loss)
        # F1 is typically -2 * H1
        self.f0 = 2 * self.h0
        self.f1 = -2 * self.h1

    def analysis(self, x):
        """
        Decomposes signal into Low and High subbands.
        Steps: Filter -> Downsample
        """
        # Low Band: Filter with H0
        low_filtered = signal.lfilter(self.h0, 1.0, x)
        # Downsample by 2 (keep even indices)
        y_low = low_filtered[0::2]

        # High Band: Filter with H1
        high_filtered = signal.lfilter(self.h1, 1.0, x)
        # Downsample by 2
        y_high = high_filtered[0::2]

        return y_low, y_high

    def synthesis(self, y_low, y_high):
        """
        Reconstructs signal from subbands.
        Steps: Upsample -> Filter -> Add
        """
        # 1. Upsample (Insert Zeros)
        # Create arrays of double length
        up_low = np.zeros(2 * len(y_low))
        up_high = np.zeros(2 * len(y_high))
        
        # Place values in even indices (0, 2, 4...)
        up_low[0::2] = y_low
        up_high[0::2] = y_high

        # 2. Filter with Synthesis Filters
        recon_low = signal.lfilter(self.f0, 1.0, up_low)
        recon_high = signal.lfilter(self.f1, 1.0, up_high)

        # 3. Combine
        x_recon = recon_low + recon_high
        
        return x_recon

# --- Main Execution ---

# 1. Generate a Test Signal
fs = 1024
t = np.linspace(0, 1, fs)
# A mix of low freq (10Hz) and high freq (300Hz)

# Generate the input signal
X_n_ap, bin_rep_h = generate_input(1, 1024)
X_n_h, bin_rep_h = generate_input(2, 1024)
original_signal = X_n_h

# 2. Initialize Filter Bank
fb = TwoChannelFilterBank(num_taps=32)

# 3. Perform Analysis
low_band, high_band = fb.analysis(original_signal)

# 4. Perform Synthesis
reconstructed_signal = fb.synthesis(low_band, high_band)

# 5. Compensate for Filter Delay (Group Delay)
# For FIR filters of length N, the total delay of Analysis + Synthesis is approx N-1 samples
delay = 32 - 1
reconstructed_shifted = np.roll(reconstructed_signal, -delay)
# Zero out the wrap-around from np.roll
reconstructed_shifted[-delay:] = 0

# --- Plotting ---
plt.figure(figsize=(12, 10))

# Plot 1: Original vs Reconstructed
plt.subplot(3, 1, 1)
plt.title("Original vs Reconstructed Signal")
plt.plot(t, original_signal, 'b', label='Original', alpha=0.6, linewidth=2)
# We trim the plot to ignore the edge artifacts
plt.plot(t[:-delay], reconstructed_shifted[:-delay], 'r--', label='Reconstructed (Delay Corrected)')
plt.legend()
plt.grid(True)

# Plot 2: Subbands
plt.subplot(3, 1, 2)
plt.title("Subbands (Downsampled)")
plt.plot(low_band, 'g', label='Low Band (y0)', alpha=0.8)
plt.plot(high_band, 'm', label='High Band (y1)', alpha=0.8)
plt.legend()
plt.grid(True)

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