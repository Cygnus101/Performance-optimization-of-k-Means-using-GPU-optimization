"""Independent float64 Lloyd reference and checks shared by every benchmark."""
import numpy as np

ITERATIONS = 20


def squared_distances(batch, centers):
    """Float64 reference distances, bounded temporary size and no N x K x D array."""
    distances = (np.sum(batch * batch, axis=1)[:, None]
                 + np.sum(centers * centers, axis=1)[None, :]
                 - 2.0 * (batch @ centers.T))
    return np.maximum(distances, 0.0)


def cpu_reference(X, initial, iterations=ITERATIONS):
    """Independent float64 Lloyd loop; lowest-index ties, retain empty centers."""
    X64 = X.astype(np.float64)
    centers = initial.astype(np.float64).copy()
    labels = np.empty(len(X), dtype=np.int32)
    for _ in range(iterations):
        for start in range(0, len(X), 1024):
            stop = start + 1024
            labels[start:stop] = squared_distances(X64[start:stop], centers).argmin(axis=1)
        for j in range(len(centers)):
            members = X64[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    objective = 0.0
    for start in range(0, len(X), 1024):
        batch = X64[start:start + 1024]
        distances = squared_distances(batch, centers)
        nearest = distances.argmin(axis=1)
        labels[start:start + len(batch)] = nearest
        # Direct residuals avoid norm/dot cancellation in the reported objective.
        objective += np.sum((batch - centers[nearest]) ** 2)
    return labels, centers, iterations, False, float(objective)


def validate_result(X, result, k, iterations=ITERATIONS, reference_inertia=None):
    """Validate final assignments and inertia outside timing; fail on divergence."""
    labels, centers, actual_iterations, converged, inertia = result
    if actual_iterations != iterations or converged:
        raise AssertionError("Expected fixed iterations without early stopping")
    if (labels.shape != (len(X),) or centers.shape != (k, X.shape[1])
            or centers.dtype != np.float32 or not np.issubdtype(labels.dtype, np.integer)):
        raise AssertionError("Incorrect output shape/dtype")
    if not np.isfinite(centers).all() or not np.isfinite(inertia):
        raise AssertionError("Nonfinite output")
    if np.any(labels < 0) or np.any(labels >= k):
        raise AssertionError("Labels outside [0, K)")
    centers64 = centers.astype(np.float64)
    objective = 0.0
    for start in range(0, len(X), 1024):
        batch = X[start:start + 1024].astype(np.float64)
        assigned = labels[start:start + len(batch)]
        selected = np.sum((batch - centers64[assigned]) ** 2, axis=1)
        minimum = squared_distances(batch, centers64).min(axis=1)
        np.testing.assert_allclose(selected, minimum, rtol=2e-5, atol=2e-5)
        objective += selected.sum()
    np.testing.assert_allclose(inertia, objective, rtol=2e-5, atol=2e-5)
    if reference_inertia is not None:
        # Different FP32 reduction orders can change near ties and later trajectories.
        np.testing.assert_allclose(objective, reference_inertia, rtol=5e-3, atol=1e-4)
    return float(objective)


