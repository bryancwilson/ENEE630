import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_input, rmse
from plot_sp import filter_bank_plots
from scipy import signal

# High-Pass and Low-Pass Filter Coefficients
GAIN = 2050
H0 = np.array([-1, 0, 3, 0, -8, 0, 21, 0, -45, 0, 91, 0, -191, 0, 643, 1024, 643, 0, -191, 0, 91, 0, -45, 0, 21, 0, -8, 0, 3, 0, -1])
H1 = np.array([-1, 0, 3, 0, -8, 0, 21, 0, -45, 0, 91, 0, -191, 0, 643, -1024, 643, 0, -191, 0, 91, 0, -45, 0, 21, 0, -8, 0, 3, 0, -1])
H0 = H0 / GAIN
H1 = H1 / GAIN

num_taps = 30
#H0 = signal.firwin(num_taps, cutoff=0.5, window='hamming')
#alternating_signs = np.ones(num_taps)
#alternating_signs[1::2] = -1  
#H1 = H0 * alternating_signs

# First Set of Synthesis Filters
F0_1 = np.array(H0)
F1_1 = -1*np.array(H1)

# Second Set of Synthesis Filters
F0_2 = np.array(H0)
F1_2 = np.array(H1)

def add_noise(signal, snr_db):
    # Calculate signal power and convert to dB
    signal_power = np.mean(np.abs(signal)**2)
    signal_power_db = 10 * np.log10(signal_power)

    # Calculate noise power in dB
    noise_power_db = signal_power_db - snr_db

    # Convert noise power from dB to linear scale
    noise_power = 10 ** (noise_power_db / 10)

    # Generate white Gaussian noise
    noise = np.sqrt(noise_power) * (np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape)) / np.sqrt(2)

    # Add noise to the original signal
    noisy_signal = signal + noise

    return noisy_signal

def calculate_MSE(input_option, snr, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1):
    fb = TwoChannelFilterBank(num_taps=32)

    sum = 0
    N = 100
    for _ in range(100):
        # Generate the input signal
        if input_option == 1:
            X_n_ap, bin_rep_h = generate_input(1, 1024)
            original_signal = X_n_ap
        elif input_option == 2:
            X_n_h, bin_rep_h = generate_input(2, 1024)
            original_signal = X_n_h

        # 2. Add Noise
        if snr >= 0:
            original_signal = add_noise(original_signal, snr)

        # 3. Perform Analysis
        low, v3 = fb.analysis(original_signal)
        low_me, v3_me = analysis_filter_block(H0, H1, original_signal)
        
        llow, v2 = fb.analysis(low)  # Second level analysis on high band
        llow_me, v2_me = analysis_filter_block(H0, H1, low)  # Second level analysis on high band

        v0, v1 = fb.analysis(llow)  # Third level analysis on hh band
        v0_me, v1_me = analysis_filter_block(H0, H1, llow)

        # Add noise to v's
        # if snr > 0:
        #     v3_me = add_noise(v3_me, snr)
        #     v2_me = add_noise(v2_me, snr)
        #     v1_me = add_noise(v1_me, snr)
        #     v0_me = add_noise(v0_me, snr)

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
    
    avg_mse = sum / 100
    print(f"Average RMSE over {N} runs: {avg_mse}, Input Type {input_option}, SNR={snr}dB")
    return avg_mse

def filter_bank_2(X_n):
    # ----- Analysis Filter Bank -----
    Y_n_level_1_top, Y_n_level_1_bot = analysis_filter_block(H0, H1, X_n)
    Y_n_level_2_top, Y_n_level_2_bot = analysis_filter_block(H0, H1, Y_n_level_1_top)
    Y_n_level_3_top, Y_n_level_3_bot = analysis_filter_block(H0, H1, Y_n_level_2_top)

    # Flatten to ensure 1D arrays
    v3 = np.array(Y_n_level_1_bot).flatten()
    v2 = np.array(Y_n_level_2_bot).flatten()
    v1 = np.array(Y_n_level_3_bot).flatten()
    v0 = np.array(Y_n_level_3_top).flatten()

    # ----- Synthesis Filter Bank -----
    x_n_rec_top = synthesis_filter(F0_1, F1_1, v0, v1)

    if len(x_n_rec_top) > len(v2):
        x_n_rec_top = x_n_rec_top[:len(v2)]

    x_n_rec_mid = synthesis_filter(F0_1, F1_1, x_n_rec_top, v2)

    if len(x_n_rec_mid) > len(v3):
        x_n_rec_mid = x_n_rec_mid[:len(v3)]

    x_n_rec = synthesis_filter(F0_1, F1_1, x_n_rec_mid, v3)

    return x_n_rec

def filter_bank(X_n, H0=H0, H1=H1, F0_1=F0_1, F1_1=F1_1):
    # ----- Analysis Filter Bank -----
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

    return x_n_rec

def analysis_filter_block(H0, H1, X_n):
    M = 2  # Decimation factor

    # 1. Create Polyphase Components (Same as before)
    # H(z) = E0(z^2) + z^-1 * E1(z^2)
    # E0 corresponds to even indices, E1 corresponds to odd indices
    E0_H0 = H0[0::M]
    E1_H0 = H0[1::M]
    E0_H1 = H1[0::M]
    E1_H1 = H1[1::M]

    # 2. Split Input (Corrected for Timing)
    # Even samples: x[0], x[2], x[4]...
    x_even = X_n[0::M]
    
    # Odd samples: Should be x[-1], x[1], x[3]... (Delayed stream)
    # Your previous code X_n[1::M] was x[1], x[3]... (Advanced stream)
    # We fix this by prepending 0 (representing x[-1]) and discarding the last element to match length
    x_odd_raw = X_n[1::M]
    x_odd = np.concatenate(([0], x_odd_raw))
    
    # Ensure lengths match exactly (handle odd/even input lengths)
    if len(x_odd) > len(x_even):
        x_odd = x_odd[:len(x_even)]
    elif len(x_odd) < len(x_even):
        x_odd = np.pad(x_odd, (0, len(x_even) - len(x_odd)), 'constant')

    # 3. Polyphase Filtering
    # Low Pass Branch (H0)
    # Y(z) = E0(z)X_even(z) + E1(z)X_odd(z)
    y_top_even = signal.lfilter(E0_H0, 1, x_even)
    y_top_odd  = signal.lfilter(E1_H0, 1, x_odd)
    Y_n_top = y_top_even + y_top_odd

    # High Pass Branch (H1)
    y_bot_even = signal.lfilter(E0_H1, 1, x_even)
    y_bot_odd  = signal.lfilter(E1_H1, 1, x_odd)
    Y_n_bot = y_bot_even + y_bot_odd

    # Return in (Low, High) order
    return Y_n_top, Y_n_bot
# def analysis_filter_block(H0, H1, X_n):
#     M = 2  # Number of phases

#     # Create polyphase components
#     E0_H0 = np.array([H0[i] for i in range(0, len(H0), M)])
#     E1_H0 = np.array([H0[i] for i in range(1, len(H0), M)])
#     E0_H1 = np.array([H1[i] for i in range(0, len(H1), M)])
#     E1_H1 = np.array([H1[i] for i in range(1, len(H1), M)])

#     # Split input into even and odd samples (This is the decimation step)
#     x_even = X_n[0::M]
    
#     # We pad with 0 if odd length to ensure even/odd arrays match size (This is the split and delay step)
#     if len(X_n) % 2 != 0:
#         x_odd = np.pad(X_n[1::M], (0, 1), 'constant')
#     else:
#         x_odd = X_n[1::M]
    
#     # (This is the filtering step and adding step)
#     # High Pass Branch
#     y_bot_even = signal.lfilter(E0_H1, 1, x_even)
#     y_bot_odd  = signal.lfilter(E1_H1, 1, x_odd)
#     Y_n_bot = y_bot_even + y_bot_odd

#     # Low Pass Branch
#     y_top_even = signal.lfilter(E0_H0, 1, x_even)
#     y_top_odd  = signal.lfilter(E1_H0, 1, x_odd)
#     Y_n_top = y_top_even + y_top_odd

#     return Y_n_bot, Y_n_top

def synthesis_filter(F0, F1, Y_n_top, Y_n_bot):
    L = 2
    
    # --- 1. Polyphase Decomposition ---
    # Low Pass Filter (F0)
    R0_F0 = F0[0::L] # Even indices
    R1_F0 = F0[1::L] # Odd indices

    # High Pass Filter (F1)
    R0_F1 = F1[0::L] # Even indices
    R1_F1 = F1[1::L] # Odd indices

    # --- 2. Filtering ---
    # Low Branch (Standard behavior: Input starts at 0)
    y_top_R0 = signal.lfilter(R0_F0, 1, Y_n_top) # Contributes to Even Output
    y_top_R1 = signal.lfilter(R1_F0, 1, Y_n_top) # Contributes to Odd Output

    # High Branch (Reference behavior: Input starts at 1)
    # Because the reference shifts inputs to [1, 3, 5], the polyphase interaction swaps.
    # The Even Output is now determined by the ODD filter coeffs (R1)
    # The Odd Output is now determined by the EVEN filter coeffs (R0)
    y_bot_R0 = signal.lfilter(R0_F1, 1, Y_n_bot) 
    y_bot_R1 = signal.lfilter(R1_F1, 1, Y_n_bot) 

    # --- 3. Combination (The Swap) ---
    
    # Even Samples: Low(R0) + High(R1)  <-- NOTE THE SWAP for High
    y_bot_R1_delayed = np.concatenate(([0], y_bot_R1[:-1]))
    x_recon_even = y_top_R0 + y_bot_R1_delayed

    # Odd Samples:  Low(R1) + High(R0)  <-- NOTE THE SWAP for High
    x_recon_odd  = y_top_R1 + y_bot_R0

    # --- 4. Interleaving ---
    x_reconstructed = np.zeros(2 * len(Y_n_top), dtype=complex)
    x_reconstructed[0::2] = x_recon_even
    x_reconstructed[1::2] = x_recon_odd
 
    return x_reconstructed * L
# def synthesis_filter(F0, F1, Y_n_top, Y_n_bot):
#     L = 2
#     # Synthesis Filter (F0) on Y_n_top (Low)
#     R0_F0 = np.array([F0[i] for i in range(0, len(F0), L)]) # Even indices of F0
#     R1_F0 = np.array([F0[i] for i in range(1, len(F0), L)]) # Odd indices of F0

#     # Synthesis Filter (F1) on Y_n_bot (High)
#     R0_F1 = np.array([F1[i] for i in range(0, len(F1), L)]) # Even indices of F1
#     R1_F1 = np.array([F1[i] for i in range(1, len(F1), L)]) # Odd indices of F1

#     # --- Filtering Step ---
#     # Filter Y_n_top (Low)
#     y_top_R0 = signal.lfilter(R0_F0, 1, Y_n_top) 
#     y_top_R1 = signal.lfilter(R1_F0, 1, Y_n_top) 

#     # Filter Y_n_bot (High)
#     y_bot_R0 = signal.lfilter(R0_F1, 1, Y_n_bot)
#     y_bot_R1 = signal.lfilter(R1_F1, 1, Y_n_bot) 

#     # --- Combination Step (Synthesis) ---
#     # Even samples of the reconstructed signal (using R0 components)
#     x_recon_even = y_top_R0 + y_bot_R0

#     # Odd samples of the reconstructed signal (using R1 components)
#     x_recon_odd = y_top_R1 + y_bot_R1

#     # --- Final Combination ---
#     x_reconstructed = np.zeros(2 * len(Y_n_top), dtype=complex) # Ensure dtype=complex

#     x_reconstructed[0::2] = x_recon_even
#     x_reconstructed[1::2] = x_recon_odd

#     return x_reconstructed * L

def analysis_filter(H0, H1, X_n, section=1):
    M = 2  # Number of phases
    L = len(H0) // M  # Length of each polyphase component

    # Create polyphase components
    E0_H0 = np.array([H0[i] for i in range(0, len(H0), M)])
    E1_H0 = np.array([H0[i] for i in range(1, len(H0), M)])
    E0_H1 = np.array([H1[i] for i in range(0, len(H1), M)])
    E1_H1 = np.array([H1[i] for i in range(1, len(H1), M)])

    # Split input into even and odd samples
    x_even = X_n[0::M]
    
    # We pad with 0 if odd length to ensure even/odd arrays match size
    if len(X_n) % 2 != 0:
        x_odd = np.pad(X_n[1::M], (0, 1), 'constant')
    else:
        x_odd = X_n[1::M]
    
    # Low Pass Branch
    y_low_even = signal.lfilter(E0_H0, 1, x_even)
    y_low_odd  = signal.lfilter(E1_H0, 1, x_odd)
    Y_n_bot = y_low_even + y_low_odd

    # High Pass Branch
    y_high_even = signal.lfilter(E0_H1, 1, x_even)
    y_high_odd  = signal.lfilter(E1_H1, 1, x_odd)
    Y_n_top = y_high_even + y_high_odd

    # recursion (decompose low branch further)
    if section == 3:
        return ['v1', [Y_n_bot]], ['v0', [Y_n_top]]
    else:
        high_tree = analysis_filter(H0, H1, Y_n_top, section=section+1)
        return high_tree, ['v'+str(3 - section + 1), [Y_n_bot]]
    
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
        self.f1 = 2 * self.h1

    def analysis(self, x):
        low_filtered = signal.lfilter(self.h0, 1.0, x)

        y_low = low_filtered[0::2]

        high_filtered = signal.lfilter(self.h1, 1.0, x)

        y_high = high_filtered[1::2]

        return y_low, y_high 

    def synthesis(self, y_low, y_high):

        up_low = np.zeros(2 * len(y_low), dtype=complex)
        up_high = np.zeros(2 * len(y_high), dtype=complex)
        
        up_low[0::2] = y_low
        up_high[1::2] = y_high

        recon_low = signal.lfilter(self.f0, 1.0, up_low)
        recon_high = signal.lfilter(self.f1, 1.0, up_high)

        x_recon = recon_low + recon_high
        
        return x_recon