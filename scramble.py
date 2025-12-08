import numpy as np

def generate_pn_sequence(pn_type: int, vector_size: int, key: int):
    # Fix 1: Initialize state with the provided key
    state = key
    output = []
    
    # 26 bit Mask (0x3FFFFFF)
    MASK_26 = 0x3FFFFFF
    
    for i in range(vector_size):
        # Extract bits for taps
        # Note: We check bits BEFORE shifting the state
        b25 = (state >> 25) & 1
        b3  = (state >> 3) & 1
        b2  = (state >> 2) & 1
        b1  = (state >> 1) & 1
        b0  = (state >> 0) & 1
        
        # Calculate Feedback Bit (XOR logic)
        if pn_type == 1:
            pn_bit = b25 ^ b3 ^ b0
        elif pn_type == 2:
            pn_bit = b25 ^ b3 ^ b2 ^ b1 ^ b0
        else:
            pn_bit = 0
            
        output.append(pn_bit)
        
        # Update State: Shift Left, apply mask, insert new bit at LSB
        state = ((state << 1) & MASK_26) | pn_bit
        
    return np.array(output)

def scramble_masker(frame_number: int, group_id: int, subband_lengths: list):
    """
    Generates the 4 pairs of masks (Real/Imag) for V0, V1, V2, V3.
    subband_lengths: List of 4 integers [len(V0), len(V1), len(V2), len(V3)]
    """
    
    # Constants from Step 1 in your first image
    shifts = [10, 11, 12, 13]
    multipliers = [1, 3, 5, 7]
    
    pnik_list = []
    pnqk_list = []
    
    # Iterate 0 to 3 for V0..V3
    for i in range(4):
        # Fix 3: Correct Formula matching Image 1
        # V_k Key = (FrameNumber << Shift) + (Multiplier * GroupID)
        k = (frame_number << shifts[i]) + (multipliers[i] * group_id)
        
        # Clip to 26 bits as per your code logic
        k = k & 0x3FFFFFF 
        
        # Generate PN Sequences
        # Note: We use the specific length required for this subband (V0..V3 differ in length)
        current_len = subband_lengths[i]
        
        real_pn = generate_pn_sequence(pn_type=1, vector_size=current_len, key=k)
        imag_pn = generate_pn_sequence(pn_type=2, vector_size=current_len, key=k)
        
        # Convert 0/1 to +1/-1 Mask
        # 0 -> 1
        # 1 -> -1
        mask_i = 1 - 2 * real_pn
        mask_q = 1 - 2 * imag_pn
        
        pnik_list.append(mask_i)
        pnqk_list.append(mask_q)
        
    return pnik_list, pnqk_list

def apply_scrambling(pnik, pnqk, input_signal):
    """
    Applies scrambling to a SINGLE signal array.
    """
    # Initialize complex output array
    scrambled_signal = np.zeros_like(input_signal, dtype=complex)
    
    # Vectorized implementation (faster than for-loop)
    # Real_out = Real_in * Mask_i
    real_part = np.real(input_signal) * pnik
    
    # Imag_out = Imag_in * Mask_q
    imag_part = np.imag(input_signal) * pnqk
    
    # Combine
    scrambled_signal = real_part + 1j * imag_part
    
    return scrambled_signal

def apply_descrambling(pnik, pnqk, scrambled_signal):
    """
    Reverses the scrambling process.
    Mathematically identical to apply_scrambling because the masks are +/- 1.
    """
    # Initialize recovered signal array
    recovered_signal = np.zeros_like(scrambled_signal, dtype=complex)
    
    # 1. Recover Real Part: Real(Rx) * Mask_i
    # If Mask_i was 1:  Val * 1 * 1 = Val
    # If Mask_i was -1: Val * -1 * -1 = Val
    real_part = np.real(scrambled_signal) * pnik
    
    # 2. Recover Imaginary Part: Imag(Rx) * Mask_q
    imag_part = np.imag(scrambled_signal) * pnqk
    
    # 3. Combine
    recovered_signal = real_part + 1j * imag_part
    
    return recovered_signal

def run_descrambler_system(scrambled_subbands, frame_num, group_id):
    """
    Main Receiver function.
    1. Re-generates the keys/masks (must match TX exactly).
    2. Descrambles each subband.
    """
    
    # Get the lengths of the received signals to generate matching masks
    lengths = [len(v) for v in scrambled_subbands]
    
    # RE-GENERATE the exact same masks used by the Transmitter
    # Note: We reuse 'scramble_masker' because the logic to create keys is identical
    masks_i, masks_q = scramble_masker(frame_num, group_id, lengths)
    
    recovered_output = []
    
    # Process each subband (V0, V1, V2, V3)
    for i in range(4):
        v_rec = apply_descrambling(masks_i[i], masks_q[i], scrambled_subbands[i])
        recovered_output.append(v_rec)
        
    return recovered_output
# --- Main Execution Block ---

def run_scrambler_system(v_subbands, frame_num, group_id):
    """
    Takes the 4 subbands (V0-V3), generates keys, and scrambles them.
    """
    # 1. Get lengths of each subband (they are different due to filter bank)
    lengths = [len(v) for v in v_subbands]
    
    # 2. Generate all Masks
    masks_i, masks_q = scramble_masker(frame_num, group_id, lengths)
    
    scrambled_output = []
    
    # 3. Apply scrambling to each subband
    for i in range(4):
        v_scrambled = apply_scrambling(masks_i[i], masks_q[i], v_subbands[i])
        scrambled_output.append(v_scrambled)
        
    return scrambled_output

# Example Usage:
# Create dummy data representing V0, V1, V2, V3 (different lengths)
v0 = np.random.randn(128) # Simulated V0
v1 = np.random.randn(128) # Simulated V1
v2 = np.random.randn(256) # Simulated V2
v3 = np.random.randn(512) # Simulated V3
original_inputs = [v0, v1, v2, v3]

# Constants
FRAME = 5
GROUP = 0x2B

# 2. Scramble (TX)
# ------------------------------------------
# (Using the function 'run_scrambler_system' from the previous answer)
tx_scrambled = run_scrambler_system(original_inputs, FRAME, GROUP)

print("Transmission Complete...")

# 3. Descramble (RX)
# ------------------------------------------
# The receiver gets 'tx_scrambled' and knows the Frame/Group
rx_recovered = run_descrambler_system(tx_scrambled, FRAME, GROUP)

# 4. Verify Accuracy
# ------------------------------------------
print("\n--- Verification ---")
for i in range(4):
    # Calculate the difference between Original and Recovered
    # We use NumPy's 'allclose' to check if they are effectively equal
    is_correct = np.allclose(original_inputs[i], rx_recovered[i])
    
    # Calculate max error (should be essentially 0, e.g., 1e-16)
    max_error = np.max(np.abs(original_inputs[i] - rx_recovered[i]))
    
    status = "SUCCESS" if is_correct else "FAIL"
    print(f"Subband V{i}: {status} (Max Error: {max_error:.2e})")

# Check specific values
print(f"\nOriginal V0[0]:  {original_inputs[0][0]:.4f}")
print(f"Recovered V0[0]: {rx_recovered[0][0]:.4f}")