import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf


def scrolling_plot(window_size, step_size, fs, time_series):
    time = np.linspace(0, time_series.shape[1] / fs, time_series.shape[1])
    start_idx = 0
    end_idx = min(window_size, time_series.shape[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    lines = []
    colors = plt.cm.get_cmap("tab10", time_series.shape[0])
    for k in range(time_series.shape[0]):
        (line,) = ax.plot(
            time[start_idx:end_idx],
            time_series[k, start_idx:end_idx],
            color=colors(k),
            label=f"Ch {k}",
        )
        lines.append(line)

    ax.set_xlabel("Time (s)")
    ax.set_xlim(time[start_idx], time[end_idx - 1])
    ax.set_ylim(np.min(time_series), np.max(time_series))
    ax.grid(True)
    ax.legend(loc="upper right")

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(
        ax_slider,
        "Scroll",
        0,
        time_series.shape[1] - window_size,
        valinit=start_idx,
        valstep=step_size,
    )

    def update(val):
        idx = int(slider.val)
        end = min(idx + window_size, time_series.shape[1])
        for k, line in enumerate(lines):
            line.set_xdata(time[idx:end])
            line.set_ydata(time_series[k, idx:end])
        ax.set_xlim(time[idx], time[end - 1])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def recurrence_plot(data, threshold=0.9):

    recurrence_matrix = abs(cosine_similarity(data.T, data.T)).astype(np.float32)

    # binarize the recurrence matrix
    recurrence_matrix = (recurrence_matrix >= threshold)
    
    # Wrap it in an xarray for Datashader
    img = xr.DataArray(recurrence_matrix, dims=["y", "x"])

    # Render a fixed-size canvas (say, 1000x1000 view)
    canvas = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = canvas.raster(img)

    # Apply shading to make it visible as an image
    image = tf.shade(agg, cmap=["white", "blue"])

    # Display the image
    image.to_pil().show()


def hypnogram_plot(probabilities, fs):

    colors = plt.cm.Set2(np.linspace(0, 1, probabilities.shape[0]))
    time = np.arange(probabilities.shape[1]) / (fs * 60)
    # Plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.stackplot(time, probabilities, colors=colors)
    ax.set_title("Hypnodensity Graph", fontsize=14)
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



def viz_rqa (input_measure, measure):

    subjects = input_measure.shape[0]
    sessions = input_measure.shape[1]
    x = np.arange(sessions)
    plt.figure(figsize=(10, 6))
    for subj in range(subjects):
        plt.plot(x, input_measure[subj, :], marker='o', label=f"Subject {subj+1}")
    plt.xlabel("Session")
    plt.ylabel(measure)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
