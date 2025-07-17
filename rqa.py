import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from torch import threshold


def calculate_recurrence_matrix(data, threshold=0.9):
    recurrence_matrix = abs(cosine_similarity(data.T, data.T)).astype(np.float32)
    # binarize the recurrence matrix
    return recurrence_matrix >= threshold


def recurrence_rate(Recurrence_matrix):

    N = Recurrence_matrix.shape[0]

    return Recurrence_matrix.sum() / N**2


def determinism(Recurrence_matrix, l_min=2):
    N = Recurrence_matrix.shape[0]
    diagline = defaultdict(int)

    for k in range(1, N):
        diag = np.diag(Recurrence_matrix, k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= 1:
                    diagline[length] += 1
                length = 0
        if length >= 1:
            diagline[length] += 1  # handle line ending at edge
    total_diag_points = sum(l * c for l, c in diagline.items())
    det_diag_points = sum(l * c for l, c in diagline.items() if l >= l_min)

    return det_diag_points / total_diag_points


def laminarity(Recurrence_matrix, v_min=2):
    N = Recurrence_matrix.shape[0]
    vertical_lines = defaultdict(int)

    for col in range(N):
        length = 0
        for row in range(N):
            if Recurrence_matrix[row, col] == 1:
                length += 1
            else:
                if length >= 1:
                    vertical_lines[length] += 1
                length = 0
        if length >= 1:
            vertical_lines[length] += 1

    total_vert_points = sum(l * c for l, c in vertical_lines.items())
    lam_vert_points = sum(l * c for l, c in vertical_lines.items() if l >= v_min)

    return lam_vert_points / total_vert_points


def trapping_time(Recurrence_matrix, v_min=2):
    N = Recurrence_matrix.shape[0]
    vertical_lines = defaultdict(int)

    for col in range(N):
        length = 0
        for row in range(N):
            if Recurrence_matrix[row, col] == 1:
                length += 1
            else:
                if length >= 1:
                    vertical_lines[length] += 1
                length = 0
        if length >= 1:
            vertical_lines[length] += 1

    total_lines = sum(c for l, c in vertical_lines.items() if l >= v_min)
    total_length = sum(l * c for l, c in vertical_lines.items() if l >= v_min)

    return total_length / total_lines
