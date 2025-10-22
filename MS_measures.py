from collections import defaultdict
from collections import Counter
from scipy.stats import entropy
import numpy as np
import torch
from mle import mle_vmf


def reorder_clusters(probabilities, kappas, mus, logalphas):
    """
    Reorder clusters based on their kappa values in descending order.
    """
    # if based on kappa only
    sorted_indices = sorted(range(len(kappas)), key=lambda i: kappas[i], reverse=True)

    # if based on weighted kappa
    # total_responsibility = probabilities.sum(axis=0)  # shape: (n_clusters,)
    # print(type(total_responsibility))
    # print(type(kappas))
    # weighted_kappa = total_responsibility.detach().numpy() * kappas
    # sorted_indices = np.argsort(weighted_kappa)[::-1]

    reordered_probabilities = [probabilities[:, i] for i in sorted_indices]
    reordered_kappas = [kappas[i] for i in sorted_indices]
    reordered_mus = [mus[i] for i in sorted_indices]
    reordered_logalphas = [logalphas[i] for i in sorted_indices]
    return reordered_probabilities, reordered_kappas, reordered_mus, reordered_logalphas


def ms_labels(probabilities, threshold):
    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()
    elif isinstance(probabilities, (list, tuple)) and all(torch.is_tensor(p) for p in probabilities):
        probabilities = np.stack([p.detach().cpu().numpy() for p in probabilities], axis=0)
    else:
        probabilities = np.array(probabilities)
    labels = []

    if threshold >= 0:
        for point in range(probabilities.shape[1]):
            max_prob = probabilities[:, point].max()
            if max_prob >= threshold:
                labels.append(probabilities[:, point].argmax() + 1)
            else:
                labels.append(0)
    else:
        labels = list(probabilities.argmax(axis=0) + 1)

    return labels


def ms_meanduration(labels):
    labels = np.array(labels)

    durations_by_state = defaultdict(list)
    current_label = labels[0]
    current_length = 1

    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_length += 1
        else:
            durations_by_state[current_label].append(current_length)
            current_label = labels[i]
            current_length = 1

    durations_by_state[current_label].append(current_length)

    # Compute mean duration for each state
    mean_durations = {
        state: sum(durations) / len(durations)
        for state, durations in durations_by_state.items()
    }

    return mean_durations, durations_by_state


def ms_occurrence_rate(labels, sampling_rate):
    occurrence_counts = defaultdict(int)
    prev_label = labels[0]

    # Start from index 1 to detect transitions
    for i in range(1, len(labels)):
        if labels[i] != prev_label:
            occurrence_counts[labels[i]] += 1
        elif i == 1:
            # If the signal starts with a valid state
            occurrence_counts[labels[0]] += 1
        prev_label = labels[i]

    # Total duration in seconds
    total_duration_sec = len(labels) / sampling_rate

    # Compute rate: occurrences per second
    occurrence_rate = {
        state: count / total_duration_sec for state, count in occurrence_counts.items()
    }

    return occurrence_rate


def ms_time_coverage(labels):
    total_points = len(labels)
    label_counts = Counter(labels)

    # Compute fraction of time each state covers
    coverage = {state: count / total_points for state, count in label_counts.items()}

    return coverage


def labels_entropy(probabilities):
    label_entropy = np.zeros(probabilities.shape[1])
    for i in range(probabilities.shape[1]):
        label_entropy[i] = entropy(probabilities[:, i])

    return label_entropy


def information_criterion(data, cluster_range=range(2, 20)):

    bic = []
    aic = []
    ric = []
    ricc = []
    ebic = []

    for number_of_clusters in cluster_range:
        mix = mle_vmf(data, number_of_clusters)

        X_tensor = torch.from_numpy(data).float()
        logliks, _ = mix(X_tensor)
        total_log_likelihood = logliks.sum().item()

        N = data.shape[0]
        M = mix.order
        d = mix.x_dim

        num_params = (M - 1) + M * (d - 1) + M

        bic.append(-2 * total_log_likelihood + num_params * np.log(N))
        aic.append(-2 * total_log_likelihood + 2 * num_params)
        ric.append(-2 * total_log_likelihood + num_params * 2 * np.log(d))
        ricc.append(
            -2 * total_log_likelihood + num_params * 2 * (np.log(d) + np.log(np.log(d)))
        )
        ebic.append(-2 * total_log_likelihood + num_params * (np.log(N) + np.log(d)))

    return bic, aic, ric, ricc, ebic
