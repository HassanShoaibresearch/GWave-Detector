import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from scipy.signal import tukey, welch, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
from scipy.linalg import lstsq
import matplotlib.ticker as mticker

plt.ion()

def read_file(filename):
    """
    Read the strain data from an HDF5 file.
    """
    try:
        dataFile = h5py.File(filename, 'r')
    except Exception as e:
        print(f"Error opening file {filename}: {e}")
        return None, None, None, None
    # Read the data quality information
    dqInfo = dataFile['quality']['simple']
    qmask = dqInfo['DQmask'][...]

    # Read the metadata
    meta = dataFile['meta']
    gpsStart = meta['GPSstart'][()]
    utc = meta['UTCstart'][()]
    duration = meta['Duration'][()]
    strain = dataFile['strain']['Strain'][()]
    dt = duration / len(strain)

    dataFile.close()
    return strain, dt, utc, gpsStart

def read_template(filename):
    """
    Read the gravitational wave template and metadata from an HDF5 file.
    """
    try:
        dataFile = h5py.File(filename, 'r')
    except Exception as e:
        print(f"Error opening template file {filename}: {e}")
        return None, None, None
    
    # Load the template (plus and cross polarizations)
    template = dataFile['template']
    tp = template[0]  # Plus polarization
    tx = template[1]  # Cross polarization
    
    # Extract metadata
    metadata = {
        "m1": dataFile["/meta"].attrs.get('m1', None),
        "m2": dataFile["/meta"].attrs.get('m2', None),
        "a1": dataFile["/meta"].attrs.get('a1', None),
        "a2": dataFile["/meta"].attrs.get('a2', None),
        "approx": dataFile["/meta"].attrs.get('approx', None)
    }
    
    dataFile.close()
    return tp, tx, metadata

def apply_window(strain, alpha=1/8):
    """
    Apply a Tukey window to the strain data to minimize spectral leakage.

    Parameters:
    - strain: The strain data array.
    - alpha: Fraction of the window inside the cosine tapered region (0 < alpha < 1).
            A smaller alpha results in a larger flat region in the center.

    Returns:
    - strain_windowed: The windowed strain data.
    """
    N = len(strain)
    window = tukey(N, alpha)
    strain_windowed = strain * window
    return strain_windowed

def compute_psd(strain_windowed, fs):
    """
    Compute the Power Spectral Density (PSD) of the windowed strain data using Welch's method.

    Parameters:
    - strain_windowed: The windowed strain data array.
    - fs: Sampling frequency of the data.

    Returns:
    - freqs: Array of sample frequencies.
    - psd: Power Spectral Density of the strain data.
    """
    # Define parameters for Welch's method
    nperseg = int(fs * 4)  # 4-second segments
    noverlap = nperseg // 2  # 50% overlap

    # Compute the PSD
    freqs, psd = welch(
        strain_windowed, fs=fs, window='hann', nperseg=nperseg,
        noverlap=noverlap, scaling='density', average='mean'
    )
    return freqs, psd

def smooth_vec(vec, npix):
    """
    Smooth the vector by convolving with a Gaussian kernel using FFT.

    Parameters:
    - vec: The input vector to be smoothed.
    - npix: The width of the Gaussian kernel in frequency bins.

    Returns:
    - smoothed_vec: The smoothed vector.
    """
    x = np.fft.fftfreq(len(vec)) * len(vec)
    gauss = np.exp(-0.5 * x**2 / npix**2)
    gauss = gauss / gauss.sum()
    smoothed_vec = np.fft.irfft(np.fft.rfft(vec) * np.fft.rfft(gauss), n=len(vec))
    return smoothed_vec

def whiten(strain, psd_interp, dt):
    """
    Whiten the strain data using the interpolated PSD.
    """
    Nt = len(strain)
    freqs = rfftfreq(Nt, dt)

    # Fourier transform of the strain data
    hf = rfft(strain)

    # Whitened frequency components
    white_hf = hf / np.sqrt(psd_interp(freqs))

    # Inverse Fourier transform to get whitened strain
    strain_whitened = irfft(white_hf, n=Nt)

    return strain_whitened

def plot_psd(freqs, psd, psd_smooth, det_name, step, event_name):
    """
    Plot the PSD at a specific processing step.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs, np.sqrt(psd), label='Original PSD', alpha=0.5)
    if psd_smooth is not None:
        plt.loglog(freqs, np.sqrt(psd_smooth), label='Smoothed PSD', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude Spectral Density [strain/$\\sqrt{\\mathrm{Hz}}$]')
    plt.title(f'{det_name} PSD after {step} for event {event_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_whitened_data(time, strain_whitened, tevent, det_name, event_name):
    """
    Plot the whitened strain data in the time domain.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time - tevent, strain_whitened, label='Whitened Strain')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Whitened Strain')
    plt.title(f'Whitened Strain Data for {det_name} - Event {event_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_whitened_fft(strain_whitened, dt, det_name, event_name):
    """
    Plot the amplitude spectrum of the whitened strain data.
    """
    Nt = len(strain_whitened)
    freqs = rfftfreq(Nt, dt)
    hf_whitened = rfft(strain_whitened)

    plt.figure(figsize=(10, 6))
    plt.loglog(freqs, np.abs(hf_whitened), label='FFT of Whitened Strain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'FFT of Whitened Strain Data for {det_name} - Event {event_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

from scipy.signal import butter, filtfilt

def bandpass_filter(data, fs, fband):
    """
    Apply a Butterworth bandpass filter to the data.

    Parameters:
    - data: The data array to be filtered.
    - fs: Sampling frequency of the data.
    - fband: Tuple or list with (low_freq, high_freq) for the bandpass.

    Returns:
    - data_bp: The bandpass-filtered data.
    """
    # Normalize the frequencies by the Nyquist frequency (fs/2)
    low = fband[0] / (0.5 * fs)
    high = fband[1] / (0.5 * fs)
    b, a = butter(4, [low, high], btype='band')
    data_bp = filtfilt(b, a, data)
    return data_bp

def matched_filter_frequency(strain, template, fs, N_template, gpsStart):
    """
    Perform matched filtering of the strain data with the template in the frequency domain.

    Parameters:
    - strain: Whitened, bandpassed strain data array.
    - template: Whitened, bandpassed template array.
    - fs: Sampling frequency.
    - N_template: Length of the original (unpadded) template.
    - gpsStart: Start time of the data.

    Returns:
    - time: Time vector corresponding to the matched filter output.
    - SNR: The SNR time series from matched filtering.
    - sigma: Normalization factor for the matched filter.
    """
    N = len(strain)
    dt = 1 / fs
    df = fs / N

    # Compute the FFT of the data and template
    data_fft = np.fft.fft(strain)
    template_fft = np.fft.fft(template)

    # Frequency bins
    freqs = np.fft.fftfreq(N, dt)

    # Since data and template are already whitened, set power_vec = 1
    power_vec = np.ones_like(freqs)

    # Perform matched filtering in frequency domain
    optimal = data_fft * np.conj(template_fft)  # power_vec =1
    optimal_time = 2 * np.fft.ifft(optimal)  # Remove fs scaling

    # Normalize the matched filter output
    sigmasq = (np.abs(template_fft)**2).sum() * df  # power_vec=1
    sigma = np.sqrt(sigmasq)
    SNR_complex = optimal_time / sigma

    # Shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = N_template // 2  # Location of peak in the template
    SNR_complex = np.roll(SNR_complex, peaksample)
    SNR = np.abs(SNR_complex)

    # Time vector
    time = np.arange(N) * dt + gpsStart

    return time, SNR, sigma

def plot_matched_filter_output(ax,time, snr, det_name, event_name):
    """
    Plot the matched filter output (SNR time series).

    Parameters:
    - time: Time vector corresponding to the SNR.
    - snr: SNR time series.
    - det_name: Name of the detector.
    - event_name: Name of the event.
    """
    ax.plot(time - tevent, snr, label=f'{det_name} Matched Filter Output')
    ax.set_xlabel('Time (s) relative to event')
    ax.set_ylabel('SNR')
    ax.legend()
    ax.grid(True)

def find_half_weight_frequency(freqs, template, psd_interp_func):
    """
    Find the frequency where half of the weighted power comes from below and half from above.

    Parameters:
    - freqs: Array of frequency bins corresponding to the template FFT.
    - template: Time-domain template signal.
    - psd_interp_func: Interpolation function for the PSD.

    Returns:
    - f_half: Frequency at which half the weighted power is below and half is above.
    """
    # Compute the Fourier transform of the template
    template_fft = rfft(template)
    
    # Interpolate the PSD to match the template's frequency bins
    psd_interp = psd_interp_func(freqs)
    
    # Handle any zero or negative PSD values to avoid division errors
    psd_interp[psd_interp <= 0] = np.inf
    
    # Compute the weighted power
    W = (np.abs(template_fft) ** 2) / psd_interp
    
    # Compute the cumulative integral of W
    cumulative_W = np.cumsum(W)
    cumulative_W /= cumulative_W[-1]  # Normalize to [0,1]
    
    # Find the frequency where cumulative_W crosses 0.5
    idx_half = np.searchsorted(cumulative_W, 0.5)
    if idx_half >= len(freqs):
        f_half = freqs[-1]
    else:
        f_half = freqs[idx_half]
    
    return f_half

def plot_cumulative_weighted_power(freqs, cumulative_power, f_half, detector_name, event_name):
    """
    Plot the cumulative weighted power and mark f_half.

    Parameters:
    - freqs: Array of frequency bins.
    - cumulative_power: Cumulative weighted power array.
    - f_half: Frequency at which cumulative power is 0.5.
    - detector_name: Name of the detector (e.g., 'L1').
    - event_name: Name of the event (e.g., 'LVT151012').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, cumulative_power, label='Cumulative Weighted Power')
    plt.axvline(x=f_half, color='red', linestyle='--', label=f'$f_{{half}}$ = {f_half:.2f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Cumulative Weighted Power')
    plt.title(f'{detector_name} Detector: Cumulative Weighted Power vs Frequency for Event {event_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_snr_near_peak(ax, time, snr, peak_idx, event_name, det_name, window_seconds=0.1):
    """
    Add the SNR around the peak for a given detector to the given Axes object.
    
    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - time: Time vector corresponding to the SNR.
    - snr: SNR time series.
    - peak_idx: Index of the SNR peak.
    - event_name: Name of the event.
    - det_name: Name of the detector ('H1' or 'L1').
    - tevent: Event time for alignment.
    - window_seconds: Duration (in seconds) before and after the peak to include in the plot.
    """
    fs = 1 / (time[1] - time[0])  # Sampling frequency based on time vector
    
    # Calculate the number of samples corresponding to the window duration
    window_samples = int(window_seconds * fs)
    
    # Determine the start and end indices for the window
    start_idx = max(peak_idx - window_samples, 0)
    end_idx = min(peak_idx + window_samples + 1, len(snr))
    
    # Extract the time and SNR data within the window
    time_window = time[start_idx:end_idx]
    snr_window = snr[start_idx:end_idx]
    
    # Add plot to the Axes
    ax.plot(time_window - tevent, snr_window, label=f'{det_name} SNR')
    ax.scatter(time[peak_idx] - tevent, snr[peak_idx], color='red', label=f'{det_name} Peak SNR')
    ax.axvline(x=time[peak_idx] - tevent, color='red', linestyle='--', label=f'{det_name} Peak Time')
    ax.set_xlabel('Time since event (s)')
    ax.set_ylabel('SNR')
    ax.set_title(f'SNR Around Peak for Event {event_name}')
    ax.legend()
    ax.grid(True)

def estimate_time_uncertainty(ax, time, snr, peak_idx, window=6):
    """
    Estimate the uncertainty in the peak time using a quadratic fit around the peak,
    and plot the results on a given Axes object.

    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - time: Time vector corresponding to the SNR.
    - snr: SNR time series.
    - peak_idx: Index of the SNR peak.
    - window: Number of samples on each side of the peak to use for the fit.

    Returns:
    - delta_t: Uncertainty in the peak time (seconds).
    """
    if peak_idx < window or peak_idx > len(snr) - window - 1:
        return np.nan  # Not enough data to fit

    # Extract data around the peak
    x = time[peak_idx - window: peak_idx + window + 1]
    y = snr[peak_idx - window: peak_idx + window + 1]
    print(f"Fitting data around peak index {peak_idx}:")
    print(f"x (time): {x}")
    print(f"y (SNR): {y}")

    # Adjust x by subtracting the first value to improve numerical stability
    offset = x[0]
    adjusted_x = x - offset
    print(f"Adjusted x (time - offset): {adjusted_x}")

    # Add plot to the given Axes
    ax.scatter(adjusted_x, y, label='Data Points', alpha=0.7)
    
    # Fit a quadratic polynomial (parabola)
    coeffs = np.polyfit(adjusted_x, y, 2)
    a, b, c = coeffs
    print(f"Coefficients: a={a}, b={b}, c={c}")

    # Generate fitted curve
    x_fit = np.linspace(adjusted_x.min(), adjusted_x.max(), 100)
    y_fit = a * x_fit**2 + b * x_fit + c
    ax.plot(x_fit, y_fit, color='red', label='Quadratic Fit')
    ax.set_xlabel('Time since peak (s)')
    ax.set_ylabel('SNR')
    ax.set_title('Quadratic Fit for Peak Uncertainty')
    ax.legend()
    ax.grid(True)

    if a == 0:
        return np.nan  # Prevent division by zero

    # Vertex of the parabola (peak position)
    t_peak = -b / (2 * a)
    print(f"Vertex (t_peak): {t_peak} s")

    # Estimate uncertainty based on curvature
    delta_t = np.sqrt(1 / (2 * abs(a))) if a != 0 else np.nan
    print(f"Time Uncertainty (delta_t): {delta_t} s")

    return delta_t


def estimate_fwhm(time, snr, peak_idx):
    """
    Estimate the Full Width at Half Maximum (FWHM) of the SNR peak.

    Parameters:
    - time: Time vector corresponding to the SNR.
    - snr: SNR time series.
    - peak_idx: Index of the SNR peak.

    Returns:
    - fwhm: Full Width at Half Maximum (seconds).
    """
    half_max = snr[peak_idx] / 2

    # Search to the left of the peak
    left_idx = peak_idx
    while left_idx > 0 and snr[left_idx] > half_max:
        left_idx -= 1

    # Linear interpolation for left crossing
    if left_idx == 0:
        t_left = time[left_idx]
    else:
        t_left = time[left_idx] + (half_max - snr[left_idx]) * (time[left_idx + 1] - time[left_idx]) / (snr[left_idx + 1] - snr[left_idx])

    # Search to the right of the peak
    right_idx = peak_idx
    while right_idx < len(snr) - 1 and snr[right_idx] > half_max:
        right_idx += 1

    # Linear interpolation for right crossing
    if right_idx == len(snr) - 1:
        t_right = time[right_idx]
    else:
        t_right = time[right_idx] + (half_max - snr[right_idx]) * (time[right_idx + 1] - time[right_idx]) / (snr[right_idx + 1] - snr[right_idx])

    # FWHM
    fwhm = t_right - t_left

    return fwhm

# ----------------- Least-Squares Fitting Functions -----------------

def calculate_ideal_snr(template, fs):
    """
    Calculate the ideal SNR for a perfectly matched filter.
    
    Parameters:
    - template: Time-domain template signal (whitened and bandpass filtered).
    - fs: Sampling frequency in Hz.

    Returns:
    - ideal_snr: Theoretical ideal SNR.
    """
    N = len(template)
    delta_f = fs / N
    A_fft = rfft(template)
    
    # Compute the power spectrum
    power_spectrum = np.abs(A_fft)**2
    
        # Compute sigma (template normalization factor)
    sigma_sq = np.sum(power_spectrum) * delta_f
    sigma = np.sqrt(sigma_sq)
    
    # Compute the unnormalized ideal SNR
    integral = 2 * np.sum(power_spectrum) * delta_f
    unnormalized_snr = np.sqrt(integral)
    
    # Normalize the ideal SNR
    ideal_snr = unnormalized_snr / sigma
    
    return ideal_snr

def least_squares_fit(data, template_p, template_x, fs, shift_range=10, use_both_polarizations=False):
    """
    Perform a least-squares fit to the data using the template(s).

    Parameters:
    - data: Array of strain data (whitened and bandpass filtered).
    - template_p: Plus polarization template (whitened and bandpass filtered).
    - template_x: Cross polarization template (whitened and bandpass filtered).
    - fs: Sampling frequency.
    - shift_range: Number of samples to shift around the best-fit shift.
    - use_both_polarizations: Boolean flag to include both polarizations.

    Returns:
    - best_shift_time: Optimal shift time in seconds.
    - best_coeffs: Coefficients [A] or [A, B].
    - min_chi2: Minimum chi-squared value.
    - improvement_chi2: Improvement in chi-squared when using both polarizations.
    - residuals: Residuals after the best fit.
    """
    N = len(data)
    min_chi2 = np.inf
    best_shift = 0
    best_coeffs = []
    chi2_single = None

    # Iterate over possible shifts
    for shift in range(-shift_range, shift_range + 1):
        if shift < 0:
            shifted_p = np.pad(template_p[:shift], (abs(shift), 0), 'constant')
            shifted_x = np.pad(template_x[:shift], (abs(shift), 0), 'constant') if use_both_polarizations else None
        elif shift > 0:
            shifted_p = np.pad(template_p[shift:], (0, shift), 'constant')
            shifted_x = np.pad(template_x[shift:], (0, shift), 'constant') if use_both_polarizations else None
        else:
            shifted_p = template_p
            shifted_x = template_x if use_both_polarizations else None

        # Truncate to match data length
        shifted_p = shifted_p[:N]
        if use_both_polarizations and shifted_x is not None:
            shifted_x = shifted_x[:N]

        # Construct design matrix
        if use_both_polarizations and shifted_x is not None:
            X = np.vstack([shifted_p, shifted_x]).T  # Shape: (N, 2)
        else:
            X = shifted_p.reshape(-1, 1)  # Shape: (N, 1)

        # Perform least-squares fit
        coeffs, residuals, rank, s = lstsq(X, data)

        # Compute chi-squared manually to avoid indexing errors
        fitted = X @ coeffs
        residuals_fit = data - fitted
        chi2 = np.sum(residuals_fit**2)

        # Update minimum chi-squared and best shift
        if chi2 < min_chi2:
            min_chi2 = chi2
            best_shift = shift
            best_coeffs = coeffs

        # Compute chi-squared for single polarization (plus) for comparison
        if chi2_single is None and use_both_polarizations:
            # Fit with only plus polarization
            X_single = shifted_p.reshape(-1, 1)
            coeffs_single, residuals_single, rank_single, s_single = lstsq(X_single, data)
            fitted_single = X_single @ coeffs_single
            residuals_single_fit = data - fitted_single
            chi2_single = np.sum(residuals_single_fit**2)

    # Improvement in chi-squared
    improvement_chi2 = chi2_single - min_chi2 if use_both_polarizations and chi2_single is not None else None

    # Convert shift to time
    best_shift_time = best_shift / fs

    # Compute residuals for the best fit
    if use_both_polarizations and template_x is not None:
        if best_shift < 0:
            shifted_p = np.pad(template_p[:best_shift], (abs(best_shift), 0), 'constant')
            shifted_x = np.pad(template_x[:best_shift], (abs(best_shift), 0), 'constant')
        elif best_shift > 0:
            shifted_p = np.pad(template_p[best_shift:], (0, best_shift), 'constant')
            shifted_x = np.pad(template_x[best_shift:], (0, best_shift), 'constant')
        else:
            shifted_p = template_p
            shifted_x = template_x

        # Truncate to match data length
        shifted_p = shifted_p[:N]
        shifted_x = shifted_x[:N]

        # Construct fitted model
        fitted = best_coeffs[0] * shifted_p + best_coeffs[1] * shifted_x
    else:
        if best_shift < 0:
            shifted_p = np.pad(template_p[:best_shift], (abs(best_shift), 0), 'constant')
        elif best_shift > 0:
            shifted_p = np.pad(template_p[best_shift:], (0, best_shift), 'constant')
        else:
            shifted_p = template_p

        # Truncate to match data length
        shifted_p = shifted_p[:N]

        # Construct fitted model
        fitted = best_coeffs[0] * shifted_p

    # Compute residuals
    residuals = data - fitted

    return best_shift_time, best_coeffs, min_chi2, improvement_chi2, residuals

def perform_least_squares_fit(detector_name, strain_whiten_bp_p, template_whiten_bp_p, template_whiten_bp_x=None, use_both_polarizations=True):
    """
    Perform least-squares fitting for a given detector.

    Parameters:
    - detector_name: 'H1' or 'L1'
    - strain_whiten_bp_p: Whitened and bandpass filtered plus polarization strain data
    - template_whiten_bp_p: Whitened and bandpass filtered plus polarization template
    - template_whiten_bp_x: Whitened and bandpass filtered cross polarization template (optional)
    - use_both_polarizations: Boolean flag to include both polarizations

    Returns:
    - best_shift_time: Best-fit time shift (s)
    - best_coeffs: Best-fit coefficients [A] or [A, B]
    - min_chi2: Minimum chi-squared value
    - improvement_chi2: Improvement in chi-squared when using both polarizations
    - residuals: Residuals after the best fit
    """
    best_shift_time, best_coeffs, min_chi2, improvement_chi2, residuals = least_squares_fit(
        strain_whiten_bp_p,
        template_whiten_bp_p,
        template_whiten_bp_x,
        fs,
        shift_range=10,
        use_both_polarizations=use_both_polarizations
    )

    print(f"\n{detector_name} Detector: Best Shift = {best_shift_time:.6f} s")
    print(f"{detector_name} Detector: Best Coefficients = {best_coeffs}")
    print(f"{detector_name} Detector: Minimum Chi-squared = {min_chi2:.2f}")
    if use_both_polarizations and improvement_chi2 is not None:
        print(f"{detector_name} Detector: χ² Improvement (Using Both Polarizations) = {improvement_chi2:.2f}")

    return best_shift_time, best_coeffs, min_chi2, improvement_chi2, residuals

def plot_fit_and_residuals(detector_name, time_vector, data, fitted, residuals, event_name):
    """
    Plot the data, fitted model, and residuals.

    Parameters:
    - detector_name: 'H1' or 'L1'
    - time_vector: Time array corresponding to the data
    - data: Original strain data
    - fitted: Fitted model data
    - residuals: Residuals after fitting
    - event_name: Name of the event
    """
    plt.figure(figsize=(14, 8))

    # Plot data and fitted model
    plt.subplot(2, 1, 1)
    plt.plot(time_vector - tevent, data, label='Data')
    plt.plot(time_vector - tevent, fitted, label='Fitted Model', color='orange')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Strain')
    plt.title(f'{detector_name} Detector: Data and Fitted Model for Event {event_name}')
    plt.legend()
    plt.grid(True)

    # Plot residuals
    plt.subplot(2, 1, 2)
    plt.plot(time_vector - tevent, residuals, label='Residuals', color='green')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Residual Strain')
    plt.title(f'{detector_name} Detector: Residuals after Fitting for Event {event_name}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ----------------- Main Processing Loop -----------------

# Read the event information from the JSON file
json_file = 'LOSC_Event_tutorial-master/BBH_events_v3.json'
try:
    with open(json_file, 'r') as f:
        events = json.load(f)
except Exception as e:
    print(f"Error reading JSON file {json_file}: {e}")
    events = {}

data_dir = 'LOSC_Event_tutorial-master/'

# Constants for angular uncertainty calculation
c = 3e5  # Speed of light in km/s
d = 3000  # Approximate separation between detectors in km (e.g., H1 and L1)

# Maximum allowable uncertainty (seconds)
MAX_UNCERTAINTY = 1  # Adjust as needed

# Loop over each event
for event_name in events:
    print(f"\nProcessing event: {event_name}")
    event = events[event_name]

    # Extract event-specific parameters
    fn_H1 = data_dir + event['fn_H1']
    fn_L1 = data_dir + event['fn_L1']
    fn_template = data_dir + event['fn_template']
    fs = event['fs']
    tevent = event['tevent']
    fband = event['fband']

    # Read the strain data for H1
    strain_H1, dt_H1, utc_H1, gpsStart_H1 = read_file(fn_H1)
    if strain_H1 is None:
        continue  # Skip to next event if file reading failed
    N_H1 = len(strain_H1)
    fs_H1 = 1.0 / dt_H1  # Sampling frequency
    time_H1 = np.arange(N_H1) * dt_H1 + gpsStart_H1  # Time vector in GPS time

    # Read the strain data for L1
    strain_L1, dt_L1, utc_L1, gpsStart_L1 = read_file(fn_L1)
    if strain_L1 is None:
        continue  # Skip to next event if file reading failed
    N_L1 = len(strain_L1)
    fs_L1 = 1.0 / dt_L1  # Sampling frequency
    time_L1 = np.arange(N_L1) * dt_L1 + gpsStart_L1  # Time vector in GPS time

    # Ensure sampling frequencies are the same
    if not (fs_H1 == fs_L1 == fs):
        print("Sampling frequencies do not match! Skipping event.")
        continue

    # Process H1 detector data
    print(f"\nProcessing H1 detector for event {event_name}")

    # Step 1: Apply window function
    strain_H1_win = apply_window(strain_H1)
    print("Applied Tukey window to H1 strain data.")

    # Step 2: Compute the PSD
    freqs_H1, psd_H1 = compute_psd(strain_H1_win, fs_H1)

    # Step 3: Smooth the PSD using smooth_vec
    bandwidth = 0.5  # Bandwidth in Hz, adjust as needed
    df_H1 = freqs_H1[1] - freqs_H1[0]  # Frequency resolution
    npix_H1 = bandwidth / df_H1  # Convert bandwidth in Hz to npix
    psd_H1_smooth = smooth_vec(psd_H1, npix_H1)
    print("Smoothed the H1 PSD using smooth_vec.")


    # Step 4: Interpolate the smoothed PSD for whitening
    psd_H1_interp = interp1d(freqs_H1, psd_H1_smooth, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Step 5: Whiten the data
    strain_H1_whiten = whiten(strain_H1_win, psd_H1_interp, dt_H1)
    print("Whitened the H1 strain data.")

   
    fig = plt.figure(figsize=(18, 10))

    # First subplot: PSD (Top)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.loglog(freqs_H1, np.sqrt(psd_H1), label='Original PSD', alpha=0.5, color='blue')
    ax1.loglog(freqs_H1, np.sqrt(psd_H1_smooth), label='Smoothed PSD', color='orange')
    ax1.set_title('H1 PSD')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude Spectral Density [strain/$\\sqrt{\\mathrm{Hz}}$]')
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Whitened Strain (Bottom Left)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=2)
    ax2.plot(time_H1 - tevent, strain_H1_whiten, label='Whitened Strain', color='blue')
    ax2.set_title('Whitened Strain Data')
    ax2.set_xlabel('Time (s) relative to event')
    ax2.set_ylabel('Whitened Strain')
    ax2.legend()
    ax2.grid(True)

    # Third subplot: FFT of Whitened Strain (Bottom Right)
    ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2, rowspan=2)
    fft_whitened_H1 = np.fft.rfft(strain_H1_whiten)
    freqs_fft_H1 = np.fft.rfftfreq(len(strain_H1_whiten), dt_H1)
    ax3.loglog(freqs_fft_H1, np.abs(fft_whitened_H1), label='FFT of Whitened Strain', color='blue')
    ax3.set_title('FFT of Whitened Strain Data')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Event: {event_name} - H1 Data Analysis", fontsize=16)
    plt.show()


    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()
    # Process L1 detector data
    print(f"\nProcessing L1 detector for event {event_name}")

    # Apply window function
    strain_L1_win = apply_window(strain_L1)
    print("Applied Tukey window to L1 strain data.")

    # Compute PSD
    freqs_L1, psd_L1 = compute_psd(strain_L1_win, fs_L1)

    # Smooth the PSD using smooth_vec
    df_L1 = freqs_L1[1] - freqs_L1[0]  # Frequency resolution
    npix_L1 = bandwidth / df_L1  # Use the same bandwidth
    psd_L1_smooth = smooth_vec(psd_L1, npix_L1)
    print("Smoothed the L1 PSD using smooth_vec.")


    # Interpolate the smoothed PSD for whitening
    psd_L1_interp = interp1d(freqs_L1, psd_L1_smooth, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Whiten the data
    strain_L1_whiten = whiten(strain_L1_win, psd_L1_interp, dt_L1)
    print("Whitened the L1 strain data.")

 
    fig = plt.figure(figsize=(18, 10))

    # First subplot: PSD (Top)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.loglog(freqs_L1, np.sqrt(psd_L1), label='Original PSD', alpha=0.5, color='blue')
    ax1.loglog(freqs_L1, np.sqrt(psd_L1_smooth), label='Smoothed PSD', color='orange')
    ax1.set_title('L1 PSD')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude Spectral Density [strain/$\\sqrt{\\mathrm{Hz}}$]')
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Whitened Strain (Bottom Left)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=2)
    ax2.plot(time_L1 - tevent, strain_L1_whiten, label='Whitened Strain', color='blue')
    ax2.set_title('Whitened Strain Data')
    ax2.set_xlabel('Time (s) relative to event')
    ax2.set_ylabel('Whitened Strain')
    ax2.legend()
    ax2.grid(True)

    # Third subplot: FFT of Whitened Strain (Bottom Right)
    ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2, rowspan=2)
    fft_whitened_L1 = np.fft.rfft(strain_L1_whiten)
    freqs_fft_L1 = np.fft.rfftfreq(len(strain_L1_whiten), dt_L1)
    ax3.loglog(freqs_fft_L1, np.abs(fft_whitened_L1), label='FFT of Whitened Strain', color='blue')
    ax3.set_title('FFT of Whitened Strain Data')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Event: {event_name} - L1 Data Analysis", fontsize=16)
    plt.show()

    # Read the template waveform and metadata
    print(f"\nReading template for event {event_name}")
    tp, tx, template_metadata = read_template(fn_template)
    if tp is None:
        print("Failed to read template. Skipping event.")
        continue

    # ----------------- Polarization Selection -----------------
    # Option to choose polarizations
    use_both_polarizations = False  # Set to False to use only plus polarization

    if use_both_polarizations:
        template = np.vstack((tp, tx))  # Shape: (2, N_template)
        print("Using both plus and cross polarizations for least-squares fitting.")
    else:
        template = tp  # Shape: (N_template,)
        print("Using only the plus polarization for least-squares fitting.")

    # Do NOT normalize the template; assume it's already correctly scaled
    if use_both_polarizations:
        template_energy_p = np.sum(template[0] ** 2)
        template_energy_x = np.sum(template[1] ** 2)
    else:
        template_energy = np.sum(template ** 2)

    # ----------------- Template Padding -----------------
    # Determine the number of samples
    N_data = len(strain_H1)  # Assuming H1 and L1 have the same length
    if use_both_polarizations:
        N_template_original = tp.shape[0]
    else:
        N_template_original = len(template)
    N_pad = N_data - N_template_original

    # Calculate the number of samples corresponding to 16 seconds
    samples_16s = int(16 * fs)  # fs is the sampling frequency

    # Pad the template to align the merger at 16 seconds
    N_prepad = samples_16s
    N_postpad = N_pad - N_prepad

    # Adjust if N_postpad is negative
    if N_postpad < 0:
        N_prepad += N_postpad
        N_postpad = 0

    # Pad the template
    if use_both_polarizations:
        template_padded = np.pad(template, ((0,0), (N_prepad, N_postpad)), 'constant')
    else:
        template_padded = np.pad(template, (N_prepad, N_postpad), 'constant')

    # Update the template and its time vector
    template = template_padded
    Nt = template.shape[1] if use_both_polarizations else len(template)
    dt_template = 1 / fs
    time_template = np.arange(Nt) * dt_template + gpsStart_H1

    # Set N_template correctly
    N_template = Nt  # Corrected from len(tp) to len(template_padded)

    # Apply windowing to the padded template
    if use_both_polarizations:
        template_win_p = apply_window(template[0])
        template_win_x = apply_window(template[1])
    else:
        template_win = apply_window(template)

    # ----------------- Whitening the Template -----------------
    # Whiten the template using the PSDs of each detector
    if use_both_polarizations:
        print("Whitening the template for H1 (Plus)")
        template_whiten_p_H1 = whiten(template_win_p, psd_H1_interp, dt_H1)
        print("Whitening the template for H1 (Cross)")
        template_whiten_x_H1 = whiten(template_win_x, psd_H1_interp, dt_H1)

        print("Whitening the template for L1 (Plus)")
        template_whiten_p_L1 = whiten(template_win_p, psd_L1_interp, dt_L1)
        print("Whitening the template for L1 (Cross)")
        template_whiten_x_L1 = whiten(template_win_x, psd_L1_interp, dt_L1)
    else:
        print("Whitening the template")
        template_whiten = whiten(template_win, psd_H1_interp, dt_H1)  # Assuming same PSD for H1 and L1

    # ----------------- Bandpass Filtering -----------------
    # Bandpass filter the whitened data and template for H1
    if use_both_polarizations:
        strain_H1_whiten_bp_p = bandpass_filter(strain_H1_whiten, fs_H1, fband)
        template_whiten_H1_bp_p = bandpass_filter(template_whiten_p_H1, fs_H1, fband)

        strain_H1_whiten_bp_x = bandpass_filter(strain_H1_whiten, fs_H1, fband)
        template_whiten_H1_bp_x = bandpass_filter(template_whiten_x_H1, fs_H1, fband)

    else:
        strain_H1_whiten_bp = bandpass_filter(strain_H1_whiten, fs_H1, fband)
        template_whiten_H1_bp = bandpass_filter(template_whiten, fs_H1, fband)

    # Bandpass filter the whitened data and template for L1
    if use_both_polarizations:
        strain_L1_whiten_bp_p = bandpass_filter(strain_L1_whiten, fs_L1, fband)
        template_whiten_L1_bp_p = bandpass_filter(template_whiten_p_L1, fs_L1, fband)

        strain_L1_whiten_bp_x = bandpass_filter(strain_L1_whiten, fs_L1, fband)
        template_whiten_L1_bp_x = bandpass_filter(template_whiten_x_L1, fs_L1, fband)
    else:
        strain_L1_whiten_bp = bandpass_filter(strain_L1_whiten, fs_L1, fband)
        template_whiten_L1_bp = bandpass_filter(template_whiten, fs_L1, fband)

    # ----------------- Matched Filtering -----------------
    # Perform frequency-domain matched filtering for H1
    print("Performing frequency-domain matched filtering for H1")
    if use_both_polarizations:
        # Combine the plus and cross templates
        combined_template_H1_bp =  np.sqrt( template_whiten_H1_bp_p**2 + template_whiten_H1_bp_x**2)  # Adjust as needed for both polarizations
        combined_strain_H1_bp =  np.sqrt( strain_H1_whiten_bp_p**2 + strain_H1_whiten_bp_x**2)
        # For simplicity, using only plus polarization here; modify if combining both
        time_H1_fd, snr_H1_fd, sigma_H1 = matched_filter_frequency(
            combined_strain_H1_bp, combined_template_H1_bp, fs_H1, N_template, gpsStart_H1)
    else:
        time_H1_fd, snr_H1_fd, sigma_H1 = matched_filter_frequency(
            strain_H1_whiten_bp, template_whiten_H1_bp, fs_H1, N_template, gpsStart_H1)

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Adjust rows/columns as needed

    # Plot and analyze H1 frequency-domain results
    plot_matched_filter_output(axs[0],time_H1_fd, snr_H1_fd, 'H1 (Freq Domain)', event_name)

    # Find the peak SNR for H1 frequency-domain method
    snr_max_H1_fd = np.max(snr_H1_fd)
    peak_idx_H1 = np.argmax(snr_H1_fd)
    time_max_H1_fd = time_H1_fd[peak_idx_H1]
    print(f"H1 (Freq Domain) Max SNR: {snr_max_H1_fd:.2f} at time {time_max_H1_fd - tevent:.4f} s relative to event")



    # Perform frequency-domain matched filtering for L1
    print("Performing frequency-domain matched filtering for L1")
    if use_both_polarizations:
        # Combine the plus and cross templates
        combined_template_L1_bp =  np.sqrt( template_whiten_L1_bp_p**2 + template_whiten_L1_bp_x**2)  # Adjust as needed for both polarizations
        combined_strain_L1_bp =  np.sqrt( strain_L1_whiten_bp_p**2 + strain_L1_whiten_bp_x**2)
        # For simplicity, using only plus polarization here; modify if combining both
        time_L1_fd, snr_L1_fd, sigma_L1 = matched_filter_frequency(
        combined_strain_L1_bp, combined_template_L1_bp, fs_H1, N_template, gpsStart_H1)
    else:
        time_L1_fd, snr_L1_fd, sigma_L1 = matched_filter_frequency(
            strain_L1_whiten_bp, template_whiten_L1_bp, fs_L1, N_template, gpsStart_L1)

    # Plot and analyze L1 frequency-domain results
    plot_matched_filter_output(axs[1],time_L1_fd, snr_L1_fd, 'L1 (Freq Domain)', event_name)

    # Find the peak SNR for L1 frequency-domain method
    snr_max_L1_fd = np.max(snr_L1_fd)
    peak_idx_L1 = np.argmax(snr_L1_fd)
    time_max_L1_fd = time_L1_fd[peak_idx_L1]
    print(f"L1 (Freq Domain) Max SNR: {snr_max_L1_fd:.2f} at time {time_max_L1_fd - tevent:.4f} s relative to event")

    # Compute Combined SNR
    snr_combined_fd = np.sqrt(snr_max_H1_fd**2 + snr_max_L1_fd**2)
    print(f"Combined H1 + L1 SNR: {snr_combined_fd:.2f}")

    # Compute Combined sigma
    sigma_combined = np.sqrt(sigma_H1**2 + sigma_L1**2)

 
    # Optionally, plot the combined SNR
    axs[2].plot(time_H1_fd - tevent, np.sqrt(snr_H1_fd**2 + snr_L1_fd**2), 'b', label='Combined H1 + L1 SNR(t)')
    axs[2].axvline(x=time_max_H1_fd - tevent, color='k', linestyle='--', label='Max Combined SNR')
    axs[2].set_xlabel('Time since event (s)')
    axs[2].set_ylabel('SNR')
    axs[2].legend()
    axs[2].grid(True)

    # --- Comparison of SNRs ---
    # Compute Analytic SNR for H1 and L1
    # Theoretical SNR is (h|s)
    if use_both_polarizations:
        # Combine the plus and cross templates

        SNR_analytic_H1 = calculate_ideal_snr(combined_template_H1_bp, fs_H1)
        SNR_analytic_L1 = calculate_ideal_snr(combined_template_L1_bp, fs_L1)  # Assuming same template for L1
        SNR_combined_analytic = np.sqrt( SNR_analytic_H1**2 + SNR_analytic_L1**2)
    else:
        SNR_analytic_H1 = calculate_ideal_snr(template_whiten_H1_bp, fs_H1)
        SNR_analytic_L1 = calculate_ideal_snr(template_whiten_L1_bp, fs_L1)  # Assuming same template for L1
        SNR_combined_analytic = np.sqrt( SNR_analytic_H1**2 + SNR_analytic_L1**2)



    print(f"\n--- SNR Comparison for Event {event_name} ---")
    print(f"H1 Observed SNR: {snr_max_H1_fd:.2f}")
    print(f"H1 Analytic SNR: {SNR_analytic_H1:.2f}")
    fractional_diff_H1 = (SNR_analytic_H1 - snr_max_H1_fd ) / snr_max_H1_fd  # Should be 0
    print(f"Fractional Difference (H1): {fractional_diff_H1:.4f}")

    print(f"L1 Observed SNR: {snr_max_L1_fd:.2f}")
    print(f"L1  SNR: {SNR_analytic_L1:.2f}")
    fractional_diff_L1 = (SNR_analytic_L1 - snr_max_L1_fd ) / snr_max_L1_fd   # Should be 0
    print(f"Fractional Difference (L1): {fractional_diff_L1:.4f}")

    print(f"Combined Observed SNR: {snr_combined_fd:.2f}")
    print(f"Combined Analytic SNR: {SNR_combined_analytic:.2f}")
    fractional_diff_combined = (SNR_combined_analytic - snr_combined_fd) / snr_combined_fd # Should be 0
    print(f"Fractional Difference (Combined): {fractional_diff_combined:.4f}")
    
    
    if use_both_polarizations:
        # After computing and whitening the template
        freqs_H1 = rfftfreq(len(combined_template_H1_bp), d=1/fs_H1)
        # Compute f_half
        f_half = find_half_weight_frequency(freqs_H1, combined_template_H1_bp, psd_H1_interp)

        # Plot the graph
        template_fft = rfft(combined_template_H1_bp)
        psd_values = psd_H1_interp(freqs_H1)
        psd_values[psd_values <= 0] = np.inf
        W = (np.abs(template_fft) ** 2) / psd_values
        cumulative_power = np.cumsum(W) / np.sum(W)
    else:
        # After computing and whitening the template
        freqs_H1 = rfftfreq(len(template_whiten_H1_bp), d=1/fs_H1)
        # Compute f_half
        f_half = find_half_weight_frequency(freqs_H1, template_whiten_H1_bp, psd_H1_interp)

        # Plot the graph
        template_fft = rfft(template_whiten_H1_bp)
        psd_values = psd_H1_interp(freqs_H1)
        psd_values[psd_values <= 0] = np.inf
        W = (np.abs(template_fft) ** 2) / psd_values
        cumulative_power = np.cumsum(W) / np.sum(W)

   
    # Plot cumulative weighted power
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_H1, cumulative_power, label='Cumulative Weighted Power', color='blue')
    plt.axvline(x=f_half, color='red', linestyle='--', label=f'$f_{{half}}$ = {f_half:.2f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Cumulative Weighted Power')
    plt.title('H1 Detector: Cumulative Weighted Power vs Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"F_half = {f_half}")
    # ----------------- Least-Squares Fitting -----------------
    print("\n--- Performing Least-Squares Fitting ---")

    # Define the template(s) for least-squares fitting
    if use_both_polarizations:
        # Use both plus and cross templates
        template_whiten_bp_p = template_whiten_H1_bp_p
        template_whiten_bp_x = template_whiten_H1_bp_x
    else:
        # Use only the plus template
        template_whiten_bp_p = template_whiten_H1_bp
        template_whiten_bp_x = None

    # Perform least-squares fit for H1
    if use_both_polarizations:
        best_shift_time_H1, best_coeffs_H1, min_chi2_H1, improvement_chi2_H1, residuals_H1 = perform_least_squares_fit(
            'H1',
            strain_H1_whiten_bp_p,  # Plus polarization
            template_whiten_bp_p,    # Plus polarization
            template_whiten_bp_x,    # Cross polarization
            use_both_polarizations=True
        )
    else:
        best_shift_time_H1, best_coeffs_H1, min_chi2_H1, improvement_chi2_H1, residuals_H1 = perform_least_squares_fit(
            'H1',
            strain_H1_whiten_bp,      # Only plus polarization
            template_whiten_bp_p,     # Only plus polarization
            template_whiten_bp_x=None,
            use_both_polarizations=False
        )

    # Perform least-squares fit for L1
    if use_both_polarizations:
        best_shift_time_L1, best_coeffs_L1, min_chi2_L1, improvement_chi2_L1, residuals_L1 = perform_least_squares_fit(
            'L1',
            strain_L1_whiten_bp_p,  # Plus polarization
            template_whiten_bp_p,    # Plus polarization
            template_whiten_bp_x,    # Cross polarization
            use_both_polarizations=True
        )
    else:
        best_shift_time_L1, best_coeffs_L1, min_chi2_L1, improvement_chi2_L1, residuals_L1 = perform_least_squares_fit(
            'L1',
            strain_L1_whiten_bp,      # Only plus polarization
            template_whiten_bp_p,     # Only plus polarization
            template_whiten_bp_x=None,
            use_both_polarizations=False
        )

    # ----------------- χ² Improvement Summary -----------------
    if use_both_polarizations:
        print(f"\n--- χ² Improvement Summary for Event {event_name} ---")
        print(f"H1 Detector: χ² Improvement = {improvement_chi2_H1:.2f}")
        print(f"L1 Detector: χ² Improvement = {improvement_chi2_L1:.2f}")
    else:
        print(f"\n--- χ² Summary for Event {event_name} ---")
        print(f"H1 Detector: Minimum χ² = {min_chi2_H1:.2f}")
        print(f"L1 Detector: Minimum χ² = {min_chi2_L1:.2f}")

    
    # ----------------- Localization and Uncertainty -----------------
    print("\n--- Step 6: Localizing Time of Arrival ---")

    # Create a figure with subplots
    fig, axs = plt.subplots(2,2, figsize=(10, 10))  # Adjust rows/columns as needed

    # Estimate uncertainty in peak time for H1
    if use_both_polarizations:
        # Find the index of the peak in the fitted model
        peak_idx_H1 = np.argmax(snr_H1_fd)
        delta_t_H1 = estimate_time_uncertainty(axs[1,0],time_H1, snr_H1_fd, peak_idx_H1)
        print(f"H1 Detector: Peak Time = {best_shift_time_H1:.4f} s relative to event")
        print(f"H1 Detector: Time Uncertainty (delta_t) = {delta_t_H1:.6f} s")
    else:
        peak_idx_H1 = np.argmax(snr_H1_fd)
        delta_t_H1 = estimate_time_uncertainty(axs[1,0],time_H1_fd, snr_H1_fd, peak_idx_H1)
        print(f"H1 Detector: Peak Time = {time_max_H1_fd - tevent:.4f} s relative to event")
        print(f"H1 Detector: Time Uncertainty (delta_t) = {delta_t_H1:.6f} s")

    # Estimate uncertainty in peak time for L1
    if use_both_polarizations:
        # Find the index of the peak in the fitted model
        peak_idx_L1 = np.argmax(snr_L1_fd)
        delta_t_L1 = estimate_time_uncertainty(axs[1,1],time_L1, snr_L1_fd, peak_idx_L1)
        print(f"L1 Detector: Peak Time = {best_shift_time_L1:.4f} s relative to event")
        print(f"L1 Detector: Time Uncertainty (delta_t) = {delta_t_L1:.6f} s")
    else:
        peak_idx_L1 = np.argmax(snr_L1_fd)
        delta_t_L1 = estimate_time_uncertainty(axs[1,1],time_L1_fd, snr_L1_fd, peak_idx_L1)
        print(f"L1 Detector: Peak Time = {time_max_L1_fd - tevent:.4f} s relative to event")
        print(f"L1 Detector: Time Uncertainty (delta_t) = {delta_t_L1:.6f} s")


    # Plot SNR around peak for H1
    if use_both_polarizations:
        plot_snr_near_peak(axs[0,0],
            time_H1,
            snr_H1_fd,
            peak_idx_H1,
            event_name,
            'H1',
            window_seconds=0.1  # Adjust the window as needed
        )
    else:
        plot_snr_near_peak(axs[0,0],
            time_H1_fd,
            snr_H1_fd,
            peak_idx_H1,
            event_name,
            'H1',
            window_seconds=0.1  # Adjust the window as needed
        )

    # Plot SNR around peak for L1
    if use_both_polarizations:
        plot_snr_near_peak(axs[0,1],
            time_L1,
            snr_L1_fd,
            peak_idx_L1,
            event_name,
            'L1',
            window_seconds=0.1  # Adjust the window as needed
        )
    else:
        plot_snr_near_peak(axs[0,1],
            time_L1_fd,
            snr_L1_fd,
            peak_idx_L1,
            event_name,
            'L1',
            window_seconds=0.1  # Adjust the window as needed
        )

    # Compute time difference and its uncertainty
    if use_both_polarizations:
        delta_t = best_shift_time_L1 - best_shift_time_H1
    else:
        delta_t = time_max_L1_fd - time_max_H1_fd
    delta_Dt = np.sqrt(delta_t_H1**2 + delta_t_L1**2)
    print(f"\nTime Difference (Delta t) between L1 and H1: {delta_t:.6f} s")
    print(f"Uncertainty in Time Difference (delta_Dt): {delta_Dt:.6f} s")


    # Print a summary of localization
    print(f"\n--- Localization Summary for Event {event_name} ---")
    print(f"Time Difference (Delta t): {delta_t:.6f} s")
    print(f"Uncertainty in Delta t (delta_Dt): {delta_Dt:.6f} s")
