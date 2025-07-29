import torch

def calculate_recurrence_matrix(data, threshold=0.9, thresholding_type = 'fixed'):
    similarity = torch.matmul(data, data.T)
    if thresholding_type == 'fixed':
        print(similarity.abs().min(), similarity.abs().max())
        recurrence_matrix = (similarity.abs() >= threshold).float()
    elif thresholding_type == 'dynamic':
        N = similarity.shape[0]
        mask = ~torch.eye(N, dtype=bool, device=similarity.device)
        sim_flat = similarity.abs()[mask]
        k = int(0.05 * sim_flat.numel())
        topk_values, _ = torch.topk(sim_flat, k)
        threshold_value = topk_values.min()

        print(f"Approximate 95th percentile threshold: {threshold_value.item():.4f}")
        recurrence_matrix = (similarity.abs() >= threshold_value).float()

    return recurrence_matrix


def recurrence_rate(R):
    N = R.shape[0]
    return R.sum().item() / (N * N)


def run_length_encoding(x):
    """
    Fast run-length encoding using PyTorch. x: 1D tensor of 0s and 1s
    Returns lengths of 1s
    """
    diff = torch.diff(x)
    run_starts = torch.where(diff == 1)[0] + 1
    run_ends = torch.where(diff == -1)[0] + 1

    if x[0] == 1:
        run_starts = torch.cat([torch.tensor([0], device=x.device), run_starts])
    if x[-1] == 1:
        run_ends = torch.cat([run_ends, torch.tensor([x.shape[0]], device=x.device)])

    run_lengths = run_ends - run_starts
    return run_lengths


def determinism(R, l_min=2):
    N = R.shape[0]
    det_total = 0
    total = 0

    for k in range(1, N):
        diag = torch.diagonal(R, offset=k)
        run_lengths = run_length_encoding(diag)
        long_runs = run_lengths[run_lengths >= l_min]
        det_total += long_runs.sum().item()
        total += run_lengths.sum().item()

    return det_total / total if total > 0 else 0


def laminarity(R, v_min=2):
    N = R.shape[0]
    lam_total = 0
    total = 0

    for col in range(N):
        col_data = R[:, col]
        run_lengths = run_length_encoding(col_data)
        long_runs = run_lengths[run_lengths >= v_min]
        lam_total += long_runs.sum().item()
        total += run_lengths.sum().item()

    return lam_total / total if total > 0 else 0


def trapping_time(R, v_min=2):
    N = R.shape[0]
    total_length = 0
    total_lines = 0

    for col in range(N):
        col_data = R[:, col]
        run_lengths = run_length_encoding(col_data)
        long_runs = run_lengths[run_lengths >= v_min]
        total_length += long_runs.sum().item()
        total_lines += len(long_runs)

    return total_length / total_lines if total_lines > 0 else 0