import numpy as np
import matplotlib.pyplot as plt
    
def generate_input(type: int, vector_size: int):
    # Paramters
    lower_bound = -1*(2**15)
    upper_bound = (2**15 - 1)
    n = np.linspace(0, vector_size - 1, vector_size)

    if type == 1:
        # Generate I and Q Component Separately
        x_i = np.random.randint(lower_bound, upper_bound, size=vector_size, dtype=np.int32)
        x_q = np.random.randint(lower_bound, upper_bound, size=vector_size, dtype=np.int32)
        
        # Combine To Form Complex Signal
        X_n = x_i + 1j * x_q

        # Plot Signal
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4))
        ax1.plot(n, X_n.real)
        ax1.set_xlabel('Samples (n)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('I Component')

        ax2.plot(n, X_n.imag)
        ax2.set_xlabel('Samples (n)')
        ax2.set_title('Q Component')

        ft = np.fft.fft(X_n)
        ft_shifted = np.fft.fftshift(ft)
        k = np.fft.fftshift(np.fft.fftfreq(vector_size, d=1)) * vector_size

        ax3.plot(k, np.abs(ft_shifted))
        ax3.set_xlabel('Frequency Index k')
        ax3.set_ylabel('Magnitude $|X[k]|$')
        ax3.set_title('3. Magnitude Spectrum (All-Pass)')
        ax3.grid(True, alpha=0.5)

        ax4.plot(k, np.angle(ft_shifted))
        ax4.set_xlabel('Frequency Index k')
        ax4.set_ylabel('Phase (radians)')
        ax4.set_title('4. Phase Spectrum (Random)')
        ax4.grid(True, alpha=0.5)
        plt.show()

    elif type == 2:
        pass

    return X_n


generate_input(1, 1024)