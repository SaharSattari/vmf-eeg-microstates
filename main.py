# %% Load data
import numpy as np

clean_EC_500 = np.load("clean_EC.npy")

# down sample to 250 Hz
#clean_EC = clean_EC_500[:, :, :, ::2]
clean_EC = clean_EC_500

# %% von Mises-Fisher clustering
from sklearn.preprocessing import normalize
from mle import mle_vmf
import torch

num_of_clusters = 4
# Initialize all_models as a nested list
all_models = [[[] for _ in range(4)] for _ in range(12)]

for subject in range(12):
    for iteration in range(4):
        if np.all(clean_EC[subject, iteration, :, :] == 0) or np.isnan(clean_EC[subject, iteration, :, :]).any():
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

        reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas = reorder_clusters(
            probabilities, kappa, mus, logalpha
        )
        for i, mu in enumerate(reordered_mus):
            ax = axes[iteration, i]
            mne.viz.plot_topomap(mu, raw_info, axes=ax, show=False, vlim=(-0.5, 0.5))
            ax.set_title(
                f"kappa: {reordered_kappas[i]:.2f}\nAlpha: {np.exp(reordered_logalphas[i]):.2f}",
                fontname="Arial",
                fontsize=15
            )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %% extract microstate measures
from MS_measures import ms_labels, ms_meanduration, ms_occurrence_rate, ms_time_coverage, reorder_clusters
import torch
from utils import extract_params

mean_duration = np.zeros((12, 4), dtype=object)
occurrence_rate = np.zeros((12, 4), dtype=object)
time_coverage = np.zeros((12, 4), dtype=object)
label_probabilities = np.zeros((12, 4), dtype=object)

final_probabilities = np.zeros((12, 4, num_of_clusters, clean_EC.shape[3]))
final_mus = np.zeros((12, 4, num_of_clusters, clean_EC.shape[2]))
final_labels = np.zeros((12, 4, clean_EC.shape[3]), dtype=int)
final_logalphas = np.zeros((12, 4, num_of_clusters))
fs = 500

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
        reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas = reorder_clusters(
            probabilities, kappa, mus, logalpha
        )
        reordered_probabilities = torch.stack(reordered_probabilities).detach().cpu().numpy()
        labels = ms_labels(reordered_probabilities, threshold=0.9)
        mean_duration[subject, iteration] = ms_meanduration(np.array(labels))
        occurrence_rate[subject, iteration] = ms_occurrence_rate(labels, fs)
        time_coverage[subject, iteration] = ms_time_coverage(labels)

        final_probabilities[subject, iteration] = reordered_probabilities
        final_mus [subject, iteration] = reordered_mus
        final_labels[subject, iteration] = labels
        final_logalphas[subject, iteration] = reordered_logalphas

#%% quantify the amount of uncertain time points
percentages = []
for subject in range(12):
    for iteration in range(4):
        indiv_label = final_labels[subject, iteration]
        # Count the number of uncertain time points (label == 0)
        uncertain_count = np.sum(indiv_label == 0)
        total_count = indiv_label.shape[0]
        uncertain_percentage = (uncertain_count / total_count) * 100
        print(f"Subject {subject + 1}, Iteration {iteration + 1}: {uncertain_percentage:.2f}% uncertain time points")
        percentages.append(uncertain_percentage)

import seaborn as sns
sns.histplot(percentages, bins=20, kde=True)
# %% entropy of labels
from MS_measures import labels_entropy, reorder_clusters
from utils import extract_params
import torch

all_entropies = []
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
        reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas = reorder_clusters(
            probabilities, kappa, mus, logalpha
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
    time_series=np.array(all_entropies[0]).reshape(1, len(all_entropies[0])),
)
#%% normalized entropy and perplexity 
import math

num_subjects = 12
num_iterations = 4
log_K = math.log(num_of_clusters)

mean_normalized_entropy = np.zeros((num_subjects, num_iterations))
perplexity = np.zeros((num_subjects, num_iterations))

for subject in range(num_subjects):
    for iteration in range(num_iterations):
        if not all_models[subject][iteration]:
            continue

        mix = all_models[subject][iteration][0]
        X = normalize(clean_EC[subject, iteration, :, :].T, norm="l2", axis=1)

        # Correct polarity
        reference_vector = X.mean(axis=0)
        for idx, row in enumerate(X):
            if np.dot(row, reference_vector) < 0:
                X[idx] = -row

        probabilities, kappa, mus, logalpha = extract_params(mix, X)
        reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas = reorder_clusters(
            probabilities, kappa, mus, logalpha
        )
        probabilities_np = torch.stack(reordered_probabilities).detach().cpu().numpy()

        # Compute entropy per time point
        epsilon = 1e-12  # for numerical stability
        H_n = -np.sum(probabilities_np * np.log(probabilities_np + epsilon), axis=0)
        H_normalized = H_n / log_K

        # Store mean normalized entropy
        mean_normalized_entropy[subject, iteration] = np.mean(H_normalized)

        # Store perplexity
        perplexity[subject, iteration] = np.exp(np.mean(H_n))
# visualize mean normalized entropy using a histogram
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))
plt.hist(mean_normalized_entropy.flatten(), bins=10, edgecolor='black')
plt.ylabel('Count', fontsize=10)
plt.title('Mean Normalized Entropy (close to 0 means low uncertainty)', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# visualize perplexity using a histogram
plt.figure(figsize=(4, 3))
plt.hist(perplexity.flatten(), bins=10, edgecolor='black')
plt.ylabel('Count', fontsize=10)
plt.title('Perplexity score (close to 1 means low uncertainty)', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# %% hypnogram visualization
from visualization import hypnogram_plot
prob = final_probabilities[0, 0, :, :]  # Example for subject 1, iteration 1
hypnogram_plot(prob, 250, [1, 2])

#%% joint probability distributions
subject = 2
iteration = 0

p_i = final_probabilities[subject, iteration, 1, :]
p_j = final_probabilities[subject, iteration, 2, :]

# Scatter plot with density
plt.scatter(p_i, p_j, s=5)
plt.xlabel('Probability of Microstate i')
plt.ylabel('Probability of Microstate j')
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
        for ms_idx in range(5):
            row = {
                "subject": subject + 1,
                "iteration": iteration + 1,
                "microstate": ms_idx,
                "duration": mean_duration[subject, iteration][0][ms_idx] if mean_duration[subject, iteration] is not None else np.nan,
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
# save df
# df.to_csv("microstate_measures.csv", index=False)
# %% exploratory data analysis of df
import seaborn as sns
from scipy.stats import pearsonr

# sns.scatterplot(data=df, x="duration", y="LAM", hue="microstate")
# sns.pairplot(df, hue="microstate")
# microstates = df["microstate"].unique()
# correlation_tables = {}

# for ms in microstates:
#     sub_df = df[df["microstate"] == ms]
#     corr = sub_df[["duration", "occurrence", "time_coverage", "RR", "DET", "LAM", "TT"]].corr(method="pearson")
#     correlation_tables[ms] = corr

#     print(f"\n📊 Correlation matrix for Microstate {ms}:\n", corr.round(3))
# Extract microstate 1 (index 1) duration and occurrence
ms1_df = df[df["microstate"] == 0]
ms1_df["duration"] = ms1_df["duration"] * 2

plt.figure(figsize=(4, 3))
sns.regplot(data=ms1_df, x="duration", y="LAM")
plt.gca().collections[1].set_alpha(0.5)  # Make the confidence interval more visible
plt.xlabel("Transition duration")
plt.ylabel("Laminarity")

# Compute Pearson r
valid = ms1_df[["duration", "LAM"]].dropna()
r, p = pearsonr(valid["duration"], valid["LAM"])
# plt.legend([f"r = 0.79"], loc="best")

plt.show()


# %% Extracting log likelihoods (goodness of fit)
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import mne
import pickle

raw_info = pickle.load(open("raw_info.pkl", "rb"))

n_states = final_probabilities.shape[2]
results = {}
subject = 10
iteration = 0

probs = final_probabilities[subject, iteration]
mus = final_mus[subject, iteration]

def time_lagged_corr(p_i, p_j, max_lag):
    # Ensure input tensors are converted to numpy arrays
    if hasattr(p_i, "detach"):
        p_i = p_i.detach().cpu().numpy()
    if hasattr(p_j, "detach"):
        p_j = p_j.detach().cpu().numpy()
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
        lags, corrs = time_lagged_corr(probs[i, :], probs[j, :], max_lag=100)
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

        # visualize corresponding microstate map
        if i == j:
            inset_ax = inset_axes(axes[i, j], width="30%", height="30%", loc='upper right')
            mne.viz.plot_topomap(mus[i], raw_info, axes=inset_ax, show=False)
            inset_ax.set_title("")  # optional: remove title

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

subject = 10
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
    mix_psd = mle_vmf(X, num_of_clusters)
    probabilitie_psd, kappa_psd, mus_psd, logalpha_psd = extract_params(mix_psd, X)
    reordered_probabilities_psd, reordered_kappas_psd, reordered_mus_psd = reorder_clusters(
        probabilitie_psd, kappa_psd, mus_psd
    )
    probs_psd = torch.stack(reordered_probabilities_psd).detach().cpu().numpy()

    for i in range(probs_psd.shape[0]):
        f, Pxx = welch(probs_psd[i, :], fs=250)
        plt.semilogy(f, Pxx, label=f'MS{i}')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectral Density of Microstate Probabilities')
    plt.xlim(0, 50)
    plt.show()
    # Visualize all mus in one figure
    plt.figure(figsize=(3 * len(mus_psd), 3))
    for i, mu in enumerate(mus_psd):
        ax = plt.subplot(1, len(mus_psd), i + 1)
        mne.viz.plot_topomap(reordered_mus_psd[i], raw_info, axes=ax, show=False)
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
from scipy.stats import pearsonr
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


#%% visualize prior probabilities, and transiiton state measures 
import matplotlib.pyplot as plt
fs = 250
# mean duration visuals
SESSION_NAMES = ['D1-AM', 'D1-PM', 'D2-AM', 'D2-PM']
UNIT_SCALE = 1000/fs
means = np.zeros((12, 4))
ci_lo = np.zeros((12, 4))
ci_hi = np.zeros((12, 4))

def mean_and_ci95(durations):
    """durations: 1D array of episode durations for one subject/session (can be empty)."""
    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        return np.nan, np.nan, np.nan  # mean, lo, hi

    durations = durations[durations != 0]
    m = durations.mean()
    sd = durations.std(ddof=1) if durations.size > 1 else 0.0
    se = sd / np.sqrt(durations.size) if durations.size > 0 else np.nan
    ci = 1.96 * se if durations.size > 1 else 0.0
    return m, m - ci, m + ci

for subject in range (12):
    for session in range (4):
        if mean_duration[subject, session] != 0:
            episodes = mean_duration[subject, session][1][0]
            #episodes = time_coverage[subject, session][0]
            m, lo, hi = mean_and_ci95(episodes)
            means[subject, session] = m * UNIT_SCALE
            ci_lo[subject, session] = lo * UNIT_SCALE
            ci_hi[subject, session] = hi * UNIT_SCALE

# ---- Plot: one line per subject with session-wise CI ribbons ----
x = np.arange(4)

plt.figure(figsize=(16, 10))
for s in range(12):
    y = means[s]
    lo = ci_lo[s]
    hi = ci_hi[s]

    # Mask out zero values (keep them empty)
    mask = y != 0
    if not np.any(mask):
        continue  # Skip subjects with all zeros

    # Only plot non-zero sessions
    plt.plot(x[mask], y[mask], marker='o' if s not in [0, 1, 7, 9, 10] else '*', linewidth=2, alpha=1, label=f'S{s+1}', markersize=10)
    plt.fill_between(x[mask], lo[mask], hi[mask], alpha=0.1)

plt.xticks(x, SESSION_NAMES, rotation=0, fontsize=20)
plt.xlabel('Session', fontname='Times New Roman', fontsize=20)
plt.ylabel('Transition duration (mean ±95% CI) [ms]', fontname='Times New Roman', fontsize=12)
plt.yticks(fontsize=20)
#plt.title('Transition-state (label = 0) duration across sessions\n(one line per subject)')
plt.grid(True, axis='y', alpha=0.3)

# If too busy, comment out legend or make it compact:
plt.legend(ncol=4, fontsize=20, frameon=False)

plt.tight_layout()
plt.show()


# %%

print(np.sum(np.array(mean_duration[0, 0][1][0]) == 1))

# %%
