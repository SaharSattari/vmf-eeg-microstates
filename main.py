# %% Load data
import numpy as np

clean_EC_500 = np.load("clean_EC.npy")

# down sample to 250 Hz
clean_EC = clean_EC_500[:, :, :, ::2]

# %% von Mises-Fisher clustering
from sklearn.preprocessing import normalize
from mle import mle_vmf

num_of_clusters = 4
# Initialize all_models as a nested list
all_models = [[[] for _ in range(4)] for _ in range(12)]

for subject in range(12):
    for iteration in range(4):
        if np.all(clean_EC[subject, iteration, :, :] == 0):
            continue
        else:
            # normalize
            X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)
            print("vector size", np.sqrt(np.sum(X[100, :] ** 2)))

            # correct topomap
            reference_vector = X.mean(axis=0)
            for idx, row in enumerate(X):
                if np.dot(row, reference_vector) < 0:
                    X[idx] = -row

            
            mix = mle_vmf(X, num_of_clusters)
            all_models[subject][iteration].append(mix)

# %% visualize mus
import pickle
import mne
import matplotlib.pyplot as plt
from utils import extract_params
from MS_measures import reorder_clusters

with open("raw_info.pkl", "rb") as f:
    raw_info = pickle.load(f)


for subject in range(12):
    fig, axes = plt.subplots(4, num_of_clusters, figsize=(12, 12))
    fig.suptitle(f"Subject {subject + 1} - All Iterations")
    plot_idx = 0
    for iteration in range(4):
        if not all_models[subject][iteration]:
            continue

        mix = all_models[subject][iteration][0]
        # normalize
        X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)
        print("vector size", np.sqrt(np.sum(X[100, :] ** 2)))

        # correct topomap
        reference_vector = X.mean(axis=0)
        for idx, row in enumerate(X):
            if np.dot(row, reference_vector) < 0:
                X[idx] = -row
        probabilities, kappa, mus = extract_params(mix, X)

        reordered_probabilities, reordered_kappas, reordered_mus = reorder_clusters(
            probabilities, kappa, mus
        )
        for i, mu in enumerate(reordered_mus):
            ax = axes[iteration, i]
            mne.viz.plot_topomap(mu, raw_info, axes=ax, show=False)
            ax.set_title(f"Iter {iteration + 1}, Mu {i + 1}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %% extract microstate measures
from MS_measures import ms_labels, ms_meanduration, ms_occurrence_rate, ms_time_coverage
import torch

mean_duration = np.zeros((12, 4), dtype=object)
occurrence_rate = np.zeros((12, 4), dtype=object)
time_coverage = np.zeros((12, 4), dtype=object)


for subject in range(12):
    for iteration in range(4):
        if not all_models[subject][iteration]:
            continue

        mix = all_models[subject][iteration][0]
        # normalize
        X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)
        print("vector size", np.sqrt(np.sum(X[100, :] ** 2)))

        # correct topomap
        reference_vector = X.mean(axis=0)
        for idx, row in enumerate(X):
            if np.dot(row, reference_vector) < 0:
                X[idx] = -row

        probabilities, kappa, mus = extract_params(mix, X)
        reordered_probabilities, reordered_kappas, reordered_mus = reorder_clusters(
            probabilities, kappa, mus
        )
        reordered_probabilities = numpy_array = (
            torch.stack(reordered_probabilities).detach().cpu().numpy()
        )
        labels = ms_labels(reordered_probabilities, threshold=0.9)
        mean_duration[subject, iteration] = ms_meanduration(np.array(labels))
        occurrence_rate[subject, iteration] = ms_occurrence_rate(labels, 250)
        time_coverage[subject, iteration] = ms_time_coverage(labels)


# %% entropy of labels
from MS_measures import labels_entropy, reorder_clusters
from utils import extract_params
import torch

all_entropies = []
for subject in range(1):
    for iteration in range(1):
        if not all_models[subject][iteration]:
            continue

        mix = all_models[subject][iteration][0]
        # normalize
        X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)
        print("vector size", np.sqrt(np.sum(X[100, :] ** 2)))

        # correct topomap
        reference_vector = X.mean(axis=0)
        for idx, row in enumerate(X):
            if np.dot(row, reference_vector) < 0:
                X[idx] = -row

        probabilities, kappa, mus = extract_params(mix, X)
        reordered_probabilities, reordered_kappas, reordered_mus = reorder_clusters(
            probabilities, kappa, mus
        )
        probabilities = torch.stack(reordered_probabilities).detach().cpu().numpy()
        label_entropy = labels_entropy(probabilities)

        all_entropies.append(label_entropy)

# %% Visualize one entorpy example
from visualization import scrolling_plot

scrolling_plot(
    window_size=500,
    step_size=500,
    fs=250,
    time_series=np.array(all_entropies[0]).reshape(1, 90000),
)
# %% hypnogram visualization
from visualization import hypnogram_plot

hypnogram_plot(probabilities, 250)


# %% Recurrence quantification analysis
from visualization import recurrence_plot
from rqa import (
    calculate_recurrence_matrix,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
)
from sklearn.preprocessing import normalize
import torch

RR = np.zeros((12, 4))
DET = np.zeros((12, 4))
LAM = np.zeros((12, 4))
TT = np.zeros((12, 4))

for subject in range(12):
    for iteration in range(4):
        if np.all(clean_EC[subject, iteration, :, :] == 0):
            continue

        # normalize
        X = clean_EC[subject, iteration, :, :] # shape: (T, C)
        X_tensor = torch.from_numpy(X).float()
        data_norm = torch.nn.functional.normalize(X_tensor.T, p=2, dim=1)  # shape: (T, C)
    
        recurrence_matrix = calculate_recurrence_matrix(data_norm, threshold=0.9)
        RR[subject, iteration] = recurrence_rate(recurrence_matrix)
        DET[subject, iteration] = determinism(recurrence_matrix, l_min=2)
        LAM[subject, iteration] = laminarity(recurrence_matrix, v_min=2)
        TT[subject, iteration] = trapping_time(recurrence_matrix)

        del recurrence_matrix

# save results in a folder called RQA_results
import os

if not os.path.exists("RQA_results"):
    os.makedirs("RQA_results")
np.save("RQA_results/RR.npy", RR)
np.save("RQA_results/DET.npy", DET)
np.save("RQA_results/LAM.npy", LAM)
np.save("RQA_results/TT.npy", TT)



# %% visualize
from visualization import viz_rqa
RR = np.load("RQA_results/RR.npy")
DET = np.load("RQA_results/DET.npy")
LAM = np.load("RQA_results/LAM.npy")
TT = np.load("RQA_results/TT.npy")

viz_rqa(RR, "Recurrence Rate")
viz_rqa(DET, "Determinism")
viz_rqa(LAM, "Laminarity")
viz_rqa(TT, "Trapping Time")

#%% Dataframe of all values 
import pandas as pd
rows = []
for subject in range(12):
    for iteration in range(4):
        if not all_models[subject][iteration]:
            continue
        # Each microstate (assume 4 microstates)
        for ms_idx in range(1,5):
            row = {
                "subject": subject + 1,
                "iteration": iteration + 1,
                "microstate": ms_idx,
                "duration": mean_duration[subject, iteration][ms_idx] if mean_duration[subject, iteration] is not None else np.nan,
                "occurrence": occurrence_rate[subject, iteration][ms_idx] if occurrence_rate[subject, iteration] is not None else np.nan,
                "time_coverage": time_coverage[subject, iteration][ms_idx] if time_coverage[subject, iteration] is not None else np.nan,
                "RR": RR[subject, iteration],
                "DET": DET[subject, iteration],
                "LAM": LAM[subject, iteration],
                "TT": TT[subject, iteration],
            }
            rows.append(row)

df = pd.DataFrame(rows)
print(df.head())

# %% scatter plot visualization
import seaborn as sns

sns.scatterplot(data=df, x="occurrence", y="LAM", hue="microstate")

# %%
