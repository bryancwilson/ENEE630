import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def plot_signal(X_n, stochastic=True, title="Signal Plot"):

    if stochastic:
        # Split X_n into equal segments
        segment_num = 32
        segment_length = len(X_n) // segment_num

        # Apply a hamming window to each segment
        window = np.hamming(segment_length)
        X_n_segmented = [X_n[i*segment_length:(i+1)*segment_length] * window for i in range(segment_num)]  
        X_n = X_n[:segment_length]
        n = np.linspace(0, len(X_n_segmented[0]) - 1, len(X_n_segmented[0]))

        # FFT
        fts = []
        for i in range(segment_num):
            fts.append(np.abs(np.fft.fft(X_n_segmented[i])))

        ft = sum(fts) / segment_num  # Average FFT

    else:
        n = np.linspace(0, len(X_n) - 1, len(X_n))
        ft = np.abs(np.fft.fft(X_n))

    ft_shifted = np.fft.fftshift(ft)
    k = np.fft.fftshift(np.fft.fftfreq(len(X_n), d=1)) * len(X_n) 

    # Plot Signal
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4))
    ax1.plot(n, X_n.real)
    ax1.set_xlabel('Samples (n)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('I Component')

    ax2.plot(n, X_n.imag)
    ax2.set_xlabel('Samples (n)')
    ax2.set_title('Q Component')

    ax3.plot(k, 20*np.log10(ft_shifted / np.max(ft_shifted) + 1e-12))
    ax3.set_xlabel('Frequency Index k')
    ax3.set_ylabel('Power (dB)')
    ax3.set_title('Magnitude Spectrum')
    if stochastic:
        ax3.set_ylim([-60, 5])
    ax3.grid(True, alpha=0.5)

    ax4.plot(k, np.angle(ft_shifted))
    ax4.set_xlabel('Frequency Index k')
    ax4.set_ylabel('Phase (radians)')
    ax4.set_title('Phase Spectrum')
    ax4.grid(True, alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_spec(ax, x, title):
    # Compute Spectrogram
    f, t_spec, Sxx = signal.spectrogram(x, 40000, nperseg=128, noverlap=120)
    
    # Plot in dB
    pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno', vmin=-80)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    return pcm
    
def plot_filter(H0, H1, filter_toggle, ax1, ax2, label):

    # Parameters
    N = 4096

    # Upsample 
    LPF_2jw = np.zeros(len(H0)*2)
    LPF_2jw[::2] = H0
    HPF_2jw = np.zeros(len(H1)*2)
    HPF_2jw[::2] = H1
    LPF_4jw = np.zeros(len(H0)*4)
    LPF_4jw[::4] = H0
    HPF_4jw = np.zeros(len(H1)*4)
    HPF_4jw[::4] = H1

    # Frequency Transforms
    ft_lpf = np.fft.fft(H0, N)
    ft_hpf = np.fft.fft(H1, N)
    ft_lpf_2jw = np.fft.fft(LPF_2jw, N)
    ft_hpf_2jw = np.fft.fft(HPF_2jw, N)
    ft_lpf_4jw = np.fft.fft(LPF_4jw, N)
    ft_hpf_4jw = np.fft.fft(HPF_4jw, N)

    # Magnitude Spectrums
    ft_mag_lpf = np.abs(ft_lpf)
    ft_mag_hpf = np.abs(ft_hpf)
    ft_mag_lpf_2jw = np.abs(ft_lpf_2jw)
    ft_mag_hpf_2jw = np.abs(ft_hpf_2jw)
    ft_mag_lpf_4jw = np.abs(ft_lpf_4jw)
    ft_mag_hpf_4jw = np.abs(ft_hpf_4jw)

    # Phase Spectrums
    ft_phase_lpf = np.unwrap(np.angle(ft_lpf))
    ft_phase_hpf = np.unwrap(np.angle(ft_hpf))
    ft_phase_lpf_2jw = np.unwrap(np.angle(ft_lpf_2jw))
    ft_phase_hpf_2jw = np.unwrap(np.angle(ft_hpf_2jw))
    ft_phase_lpf_4jw = np.unwrap(np.angle(ft_lpf_4jw))
    ft_phase_hpf_4jw = np.unwrap(np.angle(ft_hpf_4jw))

    if filter_toggle == 'LPF':
        ft_abs = ft_mag_lpf
        ft_phase = ft_phase_lpf
    elif filter_toggle == 'H3':
        ft_abs = ft_mag_hpf
        ft_phase = ft_phase_hpf
    elif filter_toggle == 'H0':
        ft_abs = ft_mag_lpf * ft_mag_lpf_2jw * ft_mag_lpf_4jw
        ft_phase = ft_phase_lpf + ft_phase_lpf_2jw + ft_phase_lpf_4jw
    elif filter_toggle == 'H1':
        ft_abs = ft_mag_lpf * ft_mag_lpf_2jw * ft_mag_hpf_4jw
        ft_phase = ft_phase_lpf + ft_phase_lpf_2jw + ft_phase_hpf_4jw
    elif filter_toggle == 'H2':
        ft_abs = ft_mag_lpf * ft_mag_hpf_2jw
        ft_phase = ft_phase_lpf + ft_phase_hpf_2jw
        

    # Magnitude
    ft_abs = (ft_abs + 1e-12) / np.max(ft_abs)
    k = np.arange(N)
    ax1.plot(k, 10*np.log10(ft_abs), label=label)

    # Phase
    ax2.plot(k, ft_phase, label=label)
    w_radians = np.linspace(0, 2*np.pi, N, endpoint=False)
    if label in ['LPF', 'H3', 'H2']:
        slope, intercept = np.polyfit(w_radians[900:1100], ft_phase[900:1100], 1)
        delay = -slope
    elif label in ['H0', 'H1']:
        slope, intercept = np.polyfit(w_radians[190:290], ft_phase[190:290], 1)
        delay = -slope 
    print(f"Estimated Group Delay for {label}: {delay:.2f} samples")

def filter_bank_plots(H0, H1, plot_subsequent=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    plot_filter(H0, H1, 'LPF', ax1, ax2, label="LPF")
    plot_filter(H0, H1, 'H3', ax1, ax2, label="H3")

    if plot_subsequent:
        plot_filter(H0, H1, 'H2', ax1, ax2, label="H2")
        plot_filter(H0, H1, 'H1', ax1, ax2, label="H1")
        plot_filter(H0, H1, 'H0', ax1, ax2, label="H0")

    # Titles and labels
    ax1.set_title("Magnitude Response")
    ax1.set_xlabel("Frequency Index k")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_ylim([-40, 5])
    ax1.grid(True, alpha=0.5)

    ax2.set_title("Phase Response")
    ax2.set_xlabel("Frequency Index k")
    ax2.set_ylabel("Phase (radians)")
    # ax2.set_ylim([-np.pi, np.pi])
    ax2.grid(True, alpha=0.5)
    # Add legends
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --- Helper Function for Magnitude Plotting ---
def plot_magnitude_response(ax, signal_data, title, color):
    # Compute Frequency Response
    # We use 1024 points for a smooth line
    # worN=1024 calculates 1024 points up to Nyquist (Pi)
    w, h = signal.freqz(signal_data, worN=1024)
    
    # Normalize Frequency (0 to 1 where 1 is Nyquist)
    freq_axis = w / np.pi
    
    # Calculate Magnitude in dB
    # + 1e-12 prevents "log(0)" errors
    mag_db = 20 * np.log10((np.abs(h) + 1e-12) / np.max(np.abs(h)))
    
    ax.plot(freq_axis, mag_db, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("Normalized Frequency (x $\pi$ rad/sample)")
    ax.grid(True, alpha=0.6)
    
    # Set standard limits to make comparison easier
    ax.set_ylim([-60, 10]) 
    ax.set_xlim([0, 1])
