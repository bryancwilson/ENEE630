from polyphase import H0, H1, analysis_filter, analysis_filter_block, synthesis_filter, F0_1, F1_1, F1_2, F0_2
from plot_sp import plot_magnitude_response
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

# --- TEST BENCH FOR POLYPHASE FILTER BANK DECOMPOSITION AND RECONSTRUCTION ---
N = 20000 # Number of samples
duration = 1.0  # seconds
f = N/duration  # Frequency
fs = f*2  # Sampling frequency
t = np.linspace(0, duration, int(fs)) # Time vector

# Create an impulse signal
impulse = np.zeros(len(H0))
impulse[0] = 100  

# Frequency Sweep (Chirp) Signal
chirp_sig = signal.chirp(t, f0=0, t1=1, f1=f, method='linear')

# Analysis Filter Bank Decomposition
Y_n_level_1_top, Y_n_level_1_bot = analysis_filter_block(H0, H1, chirp_sig)
Y_n_level_2_top, Y_n_level_2_bot = analysis_filter_block(H0, H1, Y_n_level_1_top)
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

# --- Plotting ---
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.stem(v3)
plt.title(f"v3 (Level 1 High Pass) - Length {len(v3)}")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(v2)
plt.title(f"v2 (Level 2 High Pass) - Length {len(v2)}")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.stem(v1)
plt.title(f"v1 (Level 3 High Pass) - Length {len(v1)}")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.stem(v0)
plt.title(f"v0 (Level 3 Low Pass) - Length {len(v0)}")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 3. THE TEST FUNCTION ---
def run_test(input_sig, title):
    # Analysis
    y_top, y_bot = analysis_filter_block(H0, H1, input_sig)
    
    # Synthesis
    recon_sig = synthesis_filter(F0_1, F1_1, y_top, y_bot)
    
    # Calculate Delay
    delay = 30
    
    # D. Compare
    # Extract valid region (ignoring start/end transient artifacts)
    valid_len = len(input_sig) - delay
    in_crop = input_sig[:valid_len]
    out_crop = recon_sig[delay:delay+valid_len]
    
    # Error Calculation
    error = in_crop - out_crop
    mse = np.mean(error**2)
    
    # E. Plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(input_sig, 'k', label='Input', alpha=0.5, linewidth=2)
    plt.plot(recon_sig, 'r--', label='Recon (Delayed)')
    plt.title(f"{title} - Visual Check")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
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
        # We use vmin=-100 to clip very quiet noise so the aliasing stands out
        pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno', vmin=-80)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        return pcm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Input
    plot_spec(ax1, chirp_sig, "Input Spectrogram (Clean Chirp)")

    # Plot Output
    # We ignore the startup delay transient to see the steady state behavior
    delay = 30
    valid_recon = recon_sig[delay:] 
    pcm = plot_spec(ax2, valid_recon, "Reconstructed Spectrogram")

    # Add Colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label='Power (dB)')

    plt.show()


# --- 4. EXECUTE ---
print("Running Chirp Test (Check for Aliasing)...")
run_test(chirp_sig, "Linear Chirp Test")
print("Running Impulse Test (Check for Reconstruction)...")
