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
        probabilities, kappa, mus, logalpha = extract_params(mix, X)

        reordered_probabilities, reordered_kappas, reordered_mus = reorder_clusters(
            probabilities, kappa, mus
        )
        for i, mu in enumerate(reordered_mus):
            ax = axes[iteration, i]
            mne.viz.plot_topomap(mu, raw_info, axes=ax, show=False, vlim=(-0.5, 0.5))
            ax.set_title(f"Iter {iteration + 1}, Mu {i + 1}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %% extract microstate measures
from MS_measures import ms_labels, ms_meanduration, ms_occurrence_rate, ms_time_coverage
import torch

mean_duration = np.zeros((12, 4), dtype=object)
occurrence_rate = np.zeros((12, 4), dtype=object)
time_coverage = np.zeros((12, 4), dtype=object)
label_probabilities = np.zeros((12, 4), dtype=object)


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

        probabilities, kappa, mus, logalpha = extract_params(mix, X)
        reordered_probabilities, reordered_kappas, reordered_mus = reorder_clusters(
            probabilities, kappa, mus
        )
        reordered_probabilities = numpy_array = (
            torch.stack(reordered_probabilities).detach().cpu().numpy()
        )
        labels = ms_labels(reordered_probabilities, threshold=-1)
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

        probabilities, kappa, mus, logalpha = extract_params(mix, X)
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

#%% joint probability distributions
from scipy.stats import gaussian_kde
p_i = probabilities[0,:]
p_j = probabilities[2,:]

# Scatter plot with density
plt.scatter(p_i, p_j, s=5)
plt.xlabel('Probability of Microstate i')
plt.ylabel('Probability of Microstate j')
plt.title('KDE Joint Distribution')
plt.colorbar(label='Density')
plt.show()

#%% determining the threshold for recurrence
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_similarity_histogram(data, bins=100, batch_size=1000):
    T, C = data.shape
    hist = torch.zeros(bins, device='cpu')
    bin_edges = torch.linspace(-1, 1, bins + 1)

    for i in range(0, T, batch_size):
        end_i = min(i + batch_size, T)
        batch_i = data[i:end_i]

        for j in range(i, T, batch_size):
            end_j = min(j + batch_size, T)
            batch_j = data[j:end_j]

            sim = torch.matmul(batch_i, batch_j.T)  # (B_i, B_j)

            if i == j:
                # remove diagonal and lower triangle
                mask = torch.triu(torch.ones_like(sim), diagonal=1)
                sim = sim[mask.bool()]
            else:
                sim = sim.flatten()

            hist += torch.histc(sim, bins=bins, min=-1.0, max=1.0).cpu()

    return hist.numpy(), bin_edges.numpy()

# Global histogram accumulation
global_hist = None
bins = 100

for subject in range(12):
    for iteration in range(4):
        X = clean_EC[subject, iteration, :, :]  # (32, 45000)
        X_tensor = torch.from_numpy(X).float().to(device).T  # (45000, 32)
        data_norm = torch.nn.functional.normalize(X_tensor, p=2, dim=1)

        hist, bin_edges = compute_similarity_histogram(data_norm, bins=bins)

        # Plot per subject-iteration
        plt.figure()
        plt.bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=0.02)
        plt.title(f"Subject {subject} Iter {iteration}")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.show()

        global_hist = hist if global_hist is None else global_hist + hist

        del X_tensor, data_norm
        torch.cuda.empty_cache()

# Final global histogram
plt.figure()
plt.bar((bin_edges[:-1] + bin_edges[1:]) / 2, global_hist, width=0.02)
plt.title("Global Similarity Histogram")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.show()
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
        X = clean_EC[subject, iteration, :, :] # shape: (C, T)
        X_tensor = torch.from_numpy(X).float()
        data_norm = torch.nn.functional.normalize(X_tensor.T, p=2, dim=1)  # shape: (T, C)
    
        recurrence_matrix = calculate_recurrence_matrix(data_norm, threshold=0.9, thresholding_type='dynamic')
        RR[subject, iteration] = recurrence_rate(recurrence_matrix)
        DET[subject, iteration] = determinism(recurrence_matrix, l_min=2)
        LAM[subject, iteration] = laminarity(recurrence_matrix, v_min=2)
        TT[subject, iteration] = trapping_time(recurrence_matrix)

        del recurrence_matrix

# save results in a folder called RQA_results
import os

if not os.path.exists("RQA_results_dynamic"):
    os.makedirs("RQA_results_dynamic")
np.save("RQA_results_dynamic/RR.npy", RR)
np.save("RQA_results_dynamic/DET.npy", DET)
np.save("RQA_results_dynamic/LAM.npy", LAM)
np.save("RQA_results_dynamic/TT.npy", TT)



# %% visualize
from visualization import viz_rqa
# RR = np.load("RQA_results/RR.npy")
# DET = np.load("RQA_results/DET.npy")
# LAM = np.load("RQA_results/LAM.npy")
# TT = np.load("RQA_results/TT.npy")

RR = np.load("RQA_results_dynamic/RR.npy")
DET = np.load("RQA_results_dynamic/DET.npy")
LAM = np.load("RQA_results_dynamic/LAM.npy")
TT = np.load("RQA_results_dynamic/TT.npy")

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

sns.scatterplot(data=df[df["microstate"] == 1], x="duration", y="LAM", hue="subject")

# %% Extracting log likelihoods
from sklearn.preprocessing import normalize
from MS_measures import information_criterion

bic = np.zeros((12, 4, 18))
aic = np.zeros((12, 4, 18))
ric = np.zeros((12, 4, 18))
ricc = np.zeros((12, 4, 18))
ebic = np.zeros((12, 4, 18))

for subject in range(12):
    for iteration in range (4):

        # normalize
        X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)
        # correct topomap
        reference_vector = X.mean(axis=0)
        for idx, row in enumerate(X):
            if np.dot(row, reference_vector) < 0:
                X[idx] = -row

        bic[subject, iteration], aic[subject, iteration], ric[subject, iteration], ricc[subject, iteration], ebic[subject, iteration] = information_criterion(X, cluster_range=range(2, 20))

#%% save bic, aic, ric, ricc, ebic
import os
if not os.path.exists("information_criteria"):
    os.makedirs("information_criteria")
np.save("information_criteria/bic.npy", bic)
np.save("information_criteria/aic.npy", aic)
np.save("information_criteria/ric.npy", ric)
np.save("information_criteria/ricc.npy", ricc)
np.save("information_criteria/ebic.npy", ebic)

# %% visualize information criteria
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load
bic = np.load("information_criteria/bic.npy")
aic = np.load("information_criteria/aic.npy")
ric = np.load("information_criteria/ric.npy")
ricc = np.load("information_criteria/ricc.npy")
ebic = np.load("information_criteria/ebic.npy")


plt.figure(figsize=(12, 8))
for subject in range(12):
    for iteration in range(4):
        plt.plot(range(2, 20), bic[subject, iteration], label=f'Subject {subject+1}, Iteration {iteration+1} - BIC')
        plt.plot(range(2, 20), aic[subject, iteration], label=f'Subject {subject+1}, Iteration {iteration+1} - AIC', linestyle='--')
        plt.plot(range(2, 20), ric[subject, iteration], label=f'Subject {subject+1}, Iteration {iteration+1} - RIC', linestyle=':')
        plt.plot(range(2, 20), ricc[subject, iteration], label=f'Subject {subject+1}, Iteration {iteration+1} - RICC', linestyle='-.')
        plt.plot(range(2, 20), ebic[subject, iteration], label=f'Subject {subject+1}, Iteration {iteration+1} - EBIC', linestyle='dotted')

# average across subjects and iterations
avg_bic = np.mean(bic, axis=(0, 1))
plt.figure()
plt.plot(range(2, 20), avg_bic, label='Average BIC', color='black', linewidth=2, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Value')
plt.title('Average BIC across Subjects and Iterations')
plt.xticks(range(2, 20))  # Set x-axis ticks to all numbers 2 to 19
plt.legend()
plt.grid()
plt.show()


# %% time lagged correlations between states
n_states = probabilities.shape[0]
results = {}


def time_lagged_corr(p_i, p_j, max_lag=1000):
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(p_i[:lag], p_j[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(p_i[lag:], p_j[:-lag])[0, 1]
        else:
            corr = np.corrcoef(p_i, p_j)[0, 1]
        corrs.append(corr)
    
    return lags, corrs

for i in range(n_states):
    for j in range(n_states):
        probs = probabilities.T
        lags, corrs = time_lagged_corr(probs[:, i], probs[:, j], max_lag=100)
        results[(i, j)] = corrs

# Example: plot all i→j as subplots
fig, axes = plt.subplots(n_states, n_states, figsize=(12, 10), sharex=True, sharey=True)
for i in range(n_states):
    for j in range(n_states):
        axes[i, j].plot(lags, results[(i, j)])
        axes[i, j].axhline(0, linestyle='--', color='gray', linewidth=0.5)
        if i == n_states - 1:
            axes[i, j].set_xlabel(f"MS{j}")
        if j == 0:
            axes[i, j].set_ylabel(f"MS{i}")
plt.suptitle("Time-Lagged Correlation between Microstate Probabilities")
plt.tight_layout()
plt.show()

#%% power spectral density of microstate probabilities  
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import normalize
from mle import mle_vmf
from utils import extract_params
import mne
import pywt

subject = 2
iteration = 0 

# normalize
X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)
import pickle
with open("raw_info.pkl", "rb") as f:
    raw_info = pickle.load(f)

# # correct topomap
# reference_vector = X.mean(axis=0)
# for idx, row in enumerate(X):
#     if np.dot(row, reference_vector) < 0:
#         X[idx] = -row

for num_of_clusters in range (3, 10):
    mix = mle_vmf(X, num_of_clusters)
    probabilities, kappa, mus, logalpha = extract_params(mix, X)

    probs = probabilities.detach().cpu().numpy()

    for i in range(probs.shape[1]):
        f, Pxx = welch(probs[:, i], fs=250)
        plt.semilogy(f, Pxx, label=f'MS{i}')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectral Density of Microstate Probabilities')
    plt.xlim(0, 50)
    plt.show()
    # Visualize all mus in one figure
    plt.figure(figsize=(3 * len(mus), 3))
    for i, mu in enumerate(mus):
        ax = plt.subplot(1, len(mus), i + 1)
        mne.viz.plot_topomap(mu, raw_info, axes=ax, show=False)
        ax.set_title(f"Mu {i + 1}")
    plt.suptitle("All Microstate Templates (mus)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %% testing laplace
import pickle
import mne
raw_info = pickle.load(open("raw_info.pkl", "rb"))
raw = mne.io.RawArray(clean_EC[2, 0], raw_info)

raw_csd = mne.preprocessing.compute_current_source_density(raw)

# compute bic for raw and raw_csd
from sklearn.preprocessing import normalize
from MS_measures import information_criterion
# normalize
X = normalize(clean_EC[2, 0, :, :].T, norm="l2", axis=1)
# correct topomap
reference_vector = X.mean(axis=0)
for idx, row in enumerate(X):
    if np.dot(row, reference_vector) < 0:
        X[idx] = -row

bic, aic, ric, ricc, ebic = information_criterion(X, cluster_range=range(2, 20))


X_csd = normalize(raw_csd.get_data().T, norm="l2", axis=1)
# correct topomap
reference_vector = X_csd.mean(axis=0)
for idx, row in enumerate(X_csd):
    if np.dot(row, reference_vector) < 0:
        X_csd[idx] = -row

bic_csd, aic_csd, ric_csd, ricc_csd, ebic_csd = information_criterion(X_csd, cluster_range=range(2, 20))

# visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(range(2, 20), bic, label='BIC (Raw)', marker='o')
plt.plot(range(2, 20), bic_csd, label='BIC (CSD)', marker='x')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Value')
plt.title('BIC for Raw and CSD Data')
plt.xticks(range(2, 20))  # Set x-axis ticks to all numbers 2 to 19
plt.legend()
plt.grid()
plt.show()


