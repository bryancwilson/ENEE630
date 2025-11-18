import numpy as np
import matplotlib.pyplot as plt

def plot_signal(X_n):

    n = np.linspace(0, len(X_n) - 1, len(X_n))

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
    k = np.fft.fftshift(np.fft.fftfreq(len(X_n), d=1)) * len(X_n)

    ax3.plot(k, 20*np.log10(np.abs(ft_shifted)))
    ax3.set_xlabel('Frequency Index k')
    ax3.set_ylabel('Magnitude $|X[k]|$')
    ax3.set_title('Magnitude Spectrum')
    ax3.grid(True, alpha=0.5)

    ax4.plot(k, np.angle(ft_shifted))
    ax4.set_xlabel('Frequency Index k')
    ax4.set_ylabel('Phase (radians)')
    ax4.set_title('Phase Spectrum')
    ax4.grid(True, alpha=0.5)
    plt.show()

def plot_filter(LPF, HPF, filter_toggle, ax1, ax2, ax3, label):

    ft_lpf = np.fft.fft(LPF, 2048)
    ft_hpf = np.fft.fft(HPF, 2048)

    if filter_toggle == 'LPF':
        H = LPF
        ft = ft_lpf
    elif filter_toggle == 'HPF':
        H = HPF
        ft = ft_hpf
    elif filter_toggle == 'H0':
        # H(2z)

    elif filter_toggle == 'H1':

    elif filter_toggle == 'H2':

        
    # Time Domain Plots
    n = np.arange(len(H))
    ax1.stem(n, H, label=label)

    # Frequency response
    k = np.arange(2048)

    # Magnitude
    ax2.plot(k, 20*np.log10((np.abs(ft) + 1e-12) / np.max(np.abs(ft))), label=label)

    # Phase
    ax3.plot(k, np.angle(ft), label=label)

def filter_bank_plots(HPF, LPF, plot_subsequent=True):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    plot_filter(LPF, HPF, 'LPF', ax1, ax2, ax3, label="LPF")
    plot_filter(LPF, HPF, 'HPF', ax1, ax2, ax3, label="H3")

    if plot_subsequent:
        plot_filter(LPF, HPF, 'H2', ax1, ax2, ax3, label="H2")
        #plot_filter(LPF, HPF, 'H1', ax1, ax2, ax3, label="H1")
        #plot_filter(LPF, HPF, 'H0', ax1, ax2, ax3, label="H0")

    # Titles and labels
    ax1.set_title("Filter Coefficients")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Amplitude")

    ax2.set_title("Magnitude Response")
    ax2.set_xlabel("Frequency Index k")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True, alpha=0.5)

    ax3.set_title("Phase Response")
    ax3.set_xlabel("Frequency Index k")
    ax3.set_ylabel("Phase (rad)")
    ax3.grid(True, alpha=0.5)

    # Add legends
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.tight_layout()
    plt.show()
