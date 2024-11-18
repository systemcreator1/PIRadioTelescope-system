import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter
import csv
import threading

# SDR Setup
sdr = RtlSdr()
sdr.sample_rate = 2.048e6
sdr.center_freq = 1420e6
sdr.gain = 49.6

# Globals for Control
scanning = False
recording = False
spectrogram_enabled = False

# Signal Processing Functions
def amplify_signal(signal, factor=10):
    return signal * factor

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Scanning Function
def scan():
    global scanning, recording, spectrogram_enabled

    azimuth = azimuth_slider.get()
    elevation = elevation_slider.get()
    frequencies = np.linspace(1400e6, 1430e6, 100)

    if recording:
        file = open("scan_results.csv", "a")
        writer = csv.writer(file)

    plt.figure(figsize=(10, 6))

    for freq in frequencies:
        if not scanning:
            break

        sdr.center_freq = freq
        samples = sdr.read_samples(256 * 1024)

        amplified = amplify_signal(samples, factor=15)
        filtered = butter_lowpass_filter(amplified, cutoff=0.1 * sdr.sample_rate, fs=sdr.sample_rate)

        fft_result = np.fft.fftshift(np.fft.fft(filtered))
        power = np.abs(fft_result)**2

        if recording:
            writer.writerow([azimuth, elevation, freq, np.max(power)])

        frequency_label.config(text=f"Frequency: {freq / 1e6:.2f} MHz")

        # Live Power Spectrum Plot
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(filtered), 1 / sdr.sample_rate)), power)
        plt.title(f"Power Spectrum at {freq / 1e6:.2f} MHz")
        plt.grid()

        # Optional Spectrogram
        if spectrogram_enabled:
            plt.subplot(2, 1, 2)
            plt.specgram(filtered, Fs=int(sdr.sample_rate), NFFT=1024, noverlap=512, cmap="viridis")
            plt.title("Spectrogram")
            plt.xlabel("Time")
            plt.ylabel("Frequency")

        plt.pause(0.1)

    if recording:
        file.close()

    plt.show()

# Start Scanning Function
def start_scanning():
    global scanning
    if not scanning:
        scanning = True
        threading.Thread(target=scan, daemon=True).start()

# Stop Scanning Function
def stop_scanning():
    global scanning
    scanning = False

# Toggle Recording Function
def toggle_recording():
    global recording
    recording = not recording
    record_button.config(text="Recording: ON" if recording else "Recording: OFF", bg="lightgreen" if recording else "lightgray")

# Toggle Spectrogram Function
def toggle_spectrogram():
    global spectrogram_enabled
    spectrogram_enabled = not spectrogram_enabled
    spectrogram_button.config(text="Spectrogram: ON" if spectrogram_enabled else "Spectrogram: OFF", bg="lightblue" if spectrogram_enabled else "lightgray")

# UI Setup
root = tk.Tk()
root.title("Telescope Controller")

# Display Current Frequency
frequency_label = tk.Label(root, text="Frequency: - MHz", font=("Arial", 14))
frequency_label.pack(pady=10)

# Azimuth and Elevation Sliders
azimuth_slider = tk.Scale(root, from_=0, to=180, label="Azimuth", orient=tk.HORIZONTAL)
azimuth_slider.pack(fill=tk.X, padx=20, pady=10)

elevation_slider = tk.Scale(root, from_=0, to=180, label="Elevation", orient=tk.HORIZONTAL)
elevation_slider.pack(fill=tk.X, padx=20, pady=10)

# Control Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Scanning", command=start_scanning, font=("Arial", 12), bg="lightblue")
start_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(button_frame, text="Stop Scanning", command=stop_scanning, font=("Arial", 12), bg="red")
stop_button.grid(row=0, column=1, padx=5)

record_button = tk.Button(button_frame, text="Recording: OFF", command=toggle_recording, font=("Arial", 12), bg="lightgray")
record_button.grid(row=0, column=2, padx=5)

spectrogram_button = tk.Button(button_frame, text="Spectrogram: OFF", command=toggle_spectrogram, font=("Arial", 12), bg="lightgray")
spectrogram_button.grid(row=0, column=3, padx=5)

# Cleanup on Close
def on_close():
    global scanning
    scanning = False
    sdr.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
