#%% Load raw data
import mne
import pandas as pd
from mne_icalabel import label_components
import os


# Load one CSV file from the directory
csv_dir = "EEG_CSVs_rsEC"
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
csv_path = os.path.join(csv_dir, csv_files[0])

# Read the CSV file
df = pd.read_csv(csv_path)

# Assume columns: 'time', 'Fp1', 'Fp2', ..., 'Oz'
channel_names = [col for col in df.columns if col != "time"]
data = df[channel_names].values.T/ 1e6  # shape: (n_channels, n_times)
sfreq = 500  # or set according to your data

info = mne.create_info(channel_names, sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)

# set montage
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# re-reference to average
raw.set_eeg_reference('average', projection=False) 

# remove 60 Hz line noise
raw.notch_filter(60)

# filter from 0.5 to 100 Hz
raw.filter(1, 50, fir_design="firwin")

#ic label
ica = mne.preprocessing.ICA(n_components=0.99, random_state=97, max_iter=800, method = "infomax", fit_params=dict(extended=True))
ica.fit(raw)

ic_labels = label_components(raw, ica, method="iclabel")
labels = ic_labels["labels"]
exclude_idx = [
    idx for idx, label in enumerate(labels) if label not in ["brain"]
]
ica.apply(raw, exclude=exclude_idx)

#%% exploring the filtering effect on microstates
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from mle import mle_vmf
from utils import extract_params
import torch
from MS_measures import ms_labels, ms_meanduration, ms_occurrence_rate, ms_time_coverage, reorder_clusters
import numpy as np


data = raw.get_data() 
# down sample to 250 Hz
data = data[:, ::2]

# von mises fisher clustering

num_of_clusters = 4

# normalize
X = normalize(data.T, norm="l2", axis=1)
print("vector size", np.sqrt(np.sum(X[100, :] ** 2)))

# correct topomap
reference_vector = X.mean(axis=0)
for idx, row in enumerate(X):
    if np.dot(row, reference_vector) < 0:
        X[idx] = -row

mix = mle_vmf(X, num_of_clusters)
probabilities, kappa, mus, logalpha = extract_params(mix, X)
reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas = reorder_clusters(
    probabilities, kappa, mus, logalpha
)
# viz mus
for i in range(num_of_clusters):
    mne.viz.plot_topomap(
        reordered_mus[i], raw.info, cmap="RdBu_r", contours=0, vlim=(-0.1, 0.1), sensors=True
    )

# %% try different filters and see the effect on microstate durations.
from MS_measures import ms_meanduration, ms_labels

mean_durations_not = {}
durations_by_state_not = {}

mean_durations_th = {}
durations_by_state_th = {}

for filter_band in [20, 30, 50, 80, 100]:
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Assume columns: 'time', 'Fp1', 'Fp2', ..., 'Oz'
    channel_names = [col for col in df.columns if col != "time"]
    data = df[channel_names].values.T/ 1e6  # shape: (n_channels, n_times)
    sfreq = 500  # or set according to your data

    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # set montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    # re-reference to average
    raw.set_eeg_reference('average', projection=False) 

    # remove 60 Hz line noise
    raw.notch_filter(60)

    # filter from 0.5 to 100 Hz
    raw.filter(1, filter_band, fir_design="firwin")

    #ic label
    ica = mne.preprocessing.ICA(n_components=0.99, random_state=97, max_iter=800, method = "infomax", fit_params=dict(extended=True))
    ica.fit(raw)

    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain"]
    ]
    ica.apply(raw, exclude=exclude_idx) 



    data = raw.get_data() 
    # down sample to 250 Hz
    data = data[:, ::2]

    # normalize
    X = normalize(data.T, norm="l2", axis=1)
    print("vector size", np.sqrt(np.sum(X[100, :] ** 2)))

    # correct topomap
    reference_vector = X.mean(axis=0)
    for idx, row in enumerate(X):
        if np.dot(row, reference_vector) < 0:
            X[idx] = -row

    probabilities, kappa, mus, logalpha = extract_params(mix, X)
    reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas = reorder_clusters(
        probabilities, kappa, mus, logalpha
    )

    labels_not = ms_labels(reordered_probabilities, threshold=-1)
    labels_th = ms_labels(reordered_probabilities, threshold=0.9)


    mean_durations_not[filter_band], durations_by_state_not[filter_band] = ms_meanduration(labels_not)
    mean_durations_th[filter_band], durations_by_state_th[filter_band] = ms_meanduration(labels_th)


# %% visualize

# state label = 0 transition: only for durations_by_state_th
state_label = 4


# x = filter_bands: 20, 30, 50, 80, 100
# y all values of durations_by_state_th[filter_band][state_label]

x = list(durations_by_state_not.keys())

x_flat = []
y_flat = []

for band in x:
    ys = durations_by_state_not[band][state_label]
    x_flat.extend([band] * len(ys))
    y_flat.extend(ys)

# convert categorical band values to positions and add jitter
unique_bands = sorted(durations_by_state_not.keys())
band_to_pos = {b: i for i, b in enumerate(unique_bands)}
x_pos = np.array([band_to_pos[b] for b in x_flat]) + np.random.normal(0, 0.12, size=len(x_flat))

plt.scatter(x_pos, y_flat, alpha=0.7, s=5)
plt.xticks(range(len(unique_bands)), unique_bands)
plt.xlabel('Upper bound of frequency (Hz)')
plt.ylabel('Duration (samples)')
plt.title(f'State {state_label}')
plt.show()


# %%
