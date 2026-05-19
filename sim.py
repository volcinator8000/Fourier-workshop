#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Keeps your window working properly on Arch/Hyprland
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------
# 1. Define Initial State and Time
# ---------------------------------------------------------

t = np.linspace(0, 4, 1000)
f1 = 3.0
f2 = 5.0
a1 = 100.0
a2 = 100.0

def get_signals(freq1, freq2, amp1, amp2):
    sig1 = (amp1 / 100.0) * np.cos(2 * np.pi * freq1 * t)
    sig2 = (amp2 / 100.0) * np.cos(2 * np.pi * freq2 * t)
    return sig1, sig2, sig1 + sig2

sig1, sig2, signal = get_signals(f1, f2, a1, a2)
ft_freqs = np.linspace(0, 10, 500)

def calc_fourier_transform(sig):
    # Calculates BOTH the Real (X) coordinate and the Complex Magnitude
    x_coords = []
    magnitudes = []
    for f in ft_freqs:
        wound = sig * np.exp(-2j * np.pi * f * t)
        complex_center = np.mean(wound)
        x_coords.append(np.real(complex_center))
        magnitudes.append(np.abs(complex_center))
    return x_coords, magnitudes

ft_x_coords, ft_magnitudes = calc_fourier_transform(signal)

# ---------------------------------------------------------
# 2. Setup the Figure and Grid Layout
# ---------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
# Slightly increased wspace/hspace to prevent labels from overlapping
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.5, 1.2], bottom=0.15, top=0.85, hspace=0.6, wspace=0.25)

# Top Left: Signal 1
ax_sig1 = fig.add_subplot(gs[0, 0])
line_sig1, = ax_sig1.plot(t, sig1, color='green')
ax_sig1.set_title("Signal 1")
ax_sig1.set_xlim(0, 4)
ax_sig1.set_ylim(-1.5, 1.5)

# Top Right: Signal 2
ax_sig2 = fig.add_subplot(gs[0, 1])
line_sig2, = ax_sig2.plot(t, sig2, color='orange')
ax_sig2.set_title("Signal 2")
ax_sig2.set_xlim(0, 4)
ax_sig2.set_ylim(-1.5, 1.5)

# Middle Left: Summed Signal
ax_sum = fig.add_subplot(gs[1, 0])
line_sum, = ax_sum.plot(t, signal, color='purple')
ax_sum.set_title("Combined Signal (Sum)")
ax_sum.set_xlim(0, 4)
ax_sum.set_ylim(-2.5, 2.5)

# Middle Right: Winding Machine
ax_wind = fig.add_subplot(gs[1, 1])
ax_wind.set_aspect('equal')
ax_wind.set_xlim(-2.5, 2.5)
ax_wind.set_ylim(-2.5, 2.5)
ax_wind.set_title("Winding Machine (Complex Plane)")
ax_wind.grid(True)
ax_wind.axhline(0, color='black', linewidth=0.5)
ax_wind.axvline(0, color='black', linewidth=0.5)

# Bottom Left: Fourier Transform (X-Axis / Real Part)
ax_ft_x = fig.add_subplot(gs[2, 0])
line_ft_x, = ax_ft_x.plot(ft_freqs, ft_x_coords, color='gray')
ax_ft_x.set_title("Fourier Transform (Real / X-Axis Only)")
ax_ft_x.set_xlabel("Winding Frequency (Hz)")
ax_ft_x.set_xlim(0, 10)
ax_ft_x.set_ylim(-0.8, 0.8)
ax_ft_x.axhline(0, color='black', linewidth=0.5)

# Bottom Right: Fourier Transform (Complex Magnitude)
ax_ft_mag = fig.add_subplot(gs[2, 1])
line_ft_mag, = ax_ft_mag.plot(ft_freqs, ft_magnitudes, color='black')
ax_ft_mag.set_title("True Fourier Transform (Complex Magnitude)")
ax_ft_mag.set_xlabel("Winding Frequency (Hz)")
ax_ft_mag.set_xlim(0, 10)
ax_ft_mag.set_ylim(-0.1, 1.2)
ax_ft_mag.axhline(0, color='gray', linewidth=0.5)

# ---------------------------------------------------------
# 3. Initialize Interactive Winding Machine
# ---------------------------------------------------------

init_freq = 1.0

def wind_signal(sig, time_array, wind_freq):
    wound = sig * np.exp(-2j * np.pi * wind_freq * time_array)
    return np.real(wound), np.imag(wound)

x_wound, y_wound = wind_signal(signal, t, init_freq)
cx, cy = np.mean(x_wound), np.mean(y_wound)
magnitude = np.sqrt(cx**2 + cy**2)

wound_line, = ax_wind.plot(x_wound, y_wound, color='blue', alpha=0.5, linewidth=1)
center_point, = ax_wind.plot(cx, cy, 'ro', markersize=8)

# Tracking dots for both bottom graphs
ft_dot_x, = ax_ft_x.plot([init_freq], [cx], 'ro', markersize=8)
ft_dot_mag, = ax_ft_mag.plot([init_freq], [magnitude], 'ro', markersize=8)

# ---------------------------------------------------------
# 4. Add the Controls (Slider & TextBoxes)
# ---------------------------------------------------------

ax_slider = plt.axes([0.15, 0.05, 0.55, 0.03])
freq_slider = Slider(ax=ax_slider, label='Winding Freq (Hz)', valmin=0.0, valmax=10.0, valinit=init_freq)

ax_box_wind = plt.axes([0.80, 0.045, 0.1, 0.04])
text_box_wind = TextBox(ax_box_wind, '', initial=str(init_freq))

ax_box1 = plt.axes([0.15, 0.93, 0.15, 0.03])
text_box1 = TextBox(ax_box1, 'Freq 1 (Hz): ', initial=str(f1))

ax_amp1 = plt.axes([0.15, 0.89, 0.15, 0.03])
text_amp1 = TextBox(ax_amp1, 'Intensity 1 (%): ', initial=str(a1))

ax_box2 = plt.axes([0.60, 0.93, 0.15, 0.03])
text_box2 = TextBox(ax_box2, 'Freq 2 (Hz): ', initial=str(f2))

ax_amp2 = plt.axes([0.60, 0.89, 0.15, 0.03])
text_amp2 = TextBox(ax_amp2, 'Intensity 2 (%): ', initial=str(a2))

# ---------------------------------------------------------
# 5. Interactive Update Logic
# ---------------------------------------------------------

def update_slider(val):
    freq = freq_slider.val
    # Recalculate winding machine
    new_x, new_y = wind_signal(signal, t, freq)
    new_cx, new_cy = np.mean(new_x), np.mean(new_y)
    new_magnitude = np.sqrt(new_cx**2 + new_cy**2)
    # Update visuals
    wound_line.set_data(new_x, new_y)
    center_point.set_data([new_cx], [new_cy])
    # Update both tracking dots
    ft_dot_x.set_data([freq], [new_cx])
    ft_dot_mag.set_data([freq], [new_magnitude])
    # Sync text box
    if text_box_wind.text != f"{freq:.2f}":
        text_box_wind.set_val(f"{freq:.2f}")
    fig.canvas.draw_idle()

def submit_wind_box(text):
    try:
        val = float(text)
        freq_slider.set_val(val)
    except ValueError:
        pass

def recalculate_signals(val=None):
    global f1, f2, a1, a2, signal
    try:
        f1 = float(text_box1.text)
        f2 = float(text_box2.text)
        a1 = float(text_amp1.text)
        a2 = float(text_amp2.text)
    except ValueError:
        return

    # Update Signals
    sig1, sig2, signal = get_signals(f1, f2, a1, a2)
    line_sig1.set_ydata(sig1)
    line_sig2.set_ydata(sig2)
    line_sum.set_ydata(signal)

    # Optional dynamic y-limits for better visibility
    max_sig = max(np.max(np.abs(sig1)), np.max(np.abs(sig2)), 1.0)
    ax_sig1.set_ylim(-max_sig * 1.2, max_sig * 1.2)
    ax_sig2.set_ylim(-max_sig * 1.2, max_sig * 1.2)
    ax_sum.set_ylim(-max(np.max(np.abs(signal)), 1.0) * 1.2, max(np.max(np.abs(signal)), 1.0) * 1.2)

    # Re-calculate BOTH Fourier Transforms
    new_ft_x, new_ft_mag = calc_fourier_transform(signal)
    line_ft_x.set_ydata(new_ft_x)
    line_ft_mag.set_ydata(new_ft_mag)

    # Optional dynamic y-limits for Fourier plots
    ax_ft_x.set_ylim(min(new_ft_x) * 1.2 - 0.05, max(new_ft_x) * 1.2 + 0.05)
    ax_ft_mag.set_ylim(-0.05, max(new_ft_mag) * 1.2 + 0.05)

    # Refresh slider state
    update_slider(freq_slider.val)

freq_slider.on_changed(update_slider)
text_box_wind.on_submit(submit_wind_box)
text_box1.on_submit(recalculate_signals)
text_box2.on_submit(recalculate_signals)
text_amp1.on_submit(recalculate_signals)
text_amp2.on_submit(recalculate_signals)

plt.show()
