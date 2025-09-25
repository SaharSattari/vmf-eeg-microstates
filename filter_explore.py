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
        reordered_mus[i], raw.info, cmap="RdBu_r", contours=0, vlim=(-1, 1), sensors=True
    )

# hypnodensity of probabilities
from visualization import hypnogram_plot

probabilities_np = torch.stack(reordered_probabilities).detach().cpu().numpy()

hypnogram_plot(probabilities_np, 250, [1, 2])

labels = ms_labels(probabilities_np, threshold=0)
meanduration, dur_by_state = ms_meanduration(labels)
print("Mean Duration:", meanduration)


# %% visualize durations by state
import seaborn as sns
import matplotlib.pyplot as plt

# Convert durations to seconds
dur_by_state_sec = {state: [d / 250 for d in durations] for state, durations in dur_by_state.items()}
# Prepare data for seaborn
data = []
for state, durations in dur_by_state_sec.items():
    for duration in durations:
        data.append({"State": state, "Duration (s)": duration})
df = pd.DataFrame(data)
# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x="State", y="Duration (s)", data=df)
plt.title("Microstate Durations by State")
plt.xlabel("Microstate")
plt.ylabel("Duration (seconds)")
plt.ylim(0, 0.14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
# %%
