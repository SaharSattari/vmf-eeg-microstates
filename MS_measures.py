from collections import defaultdict
from collections import Counter
from scipy.stats import entropy
import numpy as np


def reorder_clusters(probabilities, kappas, mus):
    """
    Reorder clusters based on their kappa values in descending order.
    """
    sorted_indices = sorted(range(len(kappas)), key=lambda i: kappas[i], reverse=True)
    reordered_probabilities = [probabilities[:, i] for i in sorted_indices]
    reordered_kappas = [kappas[i] for i in sorted_indices]
    reordered_mus = [mus[i] for i in sorted_indices]
    return reordered_probabilities, reordered_kappas, reordered_mus


def ms_labels(probabilities, threshold):
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
    durations_by_state = defaultdict(list)
    current_label = labels[0]
    current_length = 1

    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_length += 1
        else:
            if current_label != 0:
                durations_by_state[current_label].append(current_length)
            current_label = labels[i]
            current_length = 1

    # Handle the final segment
    if current_label != 0:
        durations_by_state[current_label].append(current_length)

    # Compute mean duration for each state
    mean_durations = {
        state: sum(durations) / len(durations)
        for state, durations in durations_by_state.items()
    }

    return mean_durations


def ms_occurrence_rate(labels, sampling_rate):
    occurrence_counts = defaultdict(int)
    prev_label = labels[0]

    # Start from index 1 to detect transitions
    for i in range(1, len(labels)):
        if labels[i] != prev_label and labels[i] != 0:
            occurrence_counts[labels[i]] += 1
        elif i == 1 and labels[0] != 0:
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
