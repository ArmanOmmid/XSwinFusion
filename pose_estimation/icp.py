import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors

def randSO3(weight=1.0):
    axis = np.random.rand(3) - 0.5
    axis /= np.linalg.norm(axis)

    angle = ((np.random.rand() * 2 * np.pi) - np.pi) * weight

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    u, _, vt = np.linalg.svd(R)
    R_ortho = np.dot(u, vt)
    if np.linalg.det(R_ortho) < 0:
        R_ortho[:, -1] *= -1

    return R_ortho

def proc(X, Y):
    """
    X = Source (P)
    Y = Target (Q)
    Indices are correspondant
    """

    X_centeroid = np.mean(X, axis=0)
    Y_centeroid = np.mean(Y, axis=0)

    X_centered = X - X_centeroid
    Y_centered = Y - Y_centeroid

    Cov = Y_centered.T @ X_centered # Row vectors

    U, S, VT = np.linalg.svd(Cov)

    R = U @ VT
    if np.linalg.det(R) < 0:
        VT[-1, :] *= -1
        R = U @ VT

    t = Y_centeroid - (R @ X_centeroid)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def icp(X, Y, attempts=10, threshold=0.00001, max_iterations=200, index_targets=True):
    """Iterative closest point.

    Args:
        source_pcd (np.ndarray): [N1, 3]
        target_pcd (np.ndarray): [N2, 3]

    Returns:
        np.ndarray: [4, 4] rigid transformation to align source to target.
    """

    if index_targets:
        source_pcd = Y
        target_pcd = X # Oversampled; try to match with this
    else:
        source_pcd = X
        target_pcd = Y

    # Implement your own algorithm here.
    P = np.copy(source_pcd)

    KNN = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(target_pcd)

    best_error = np.inf
    best_pose = np.eye(4)
    for a in range(attempts):
        T = np.eye(4)
        T[:3, :3] = randSO3()

        last_error = np.inf
        for i in range(max_iterations):
            R = T[:3, :3]
            t = T[:3, 3]

            P = (source_pcd @ R.T) + t
            distances, indices = KNN.kneighbors(P)

            reorder = indices.ravel()

            if index_targets:
                RE = target_pcd[reorder]
                T = proc(source_pcd, RE)
            else:
                RE = source_pcd[reorder]
                T = proc(RE, target_pcd)

            error = np.sum(distances)
            if np.abs(error - last_error) < threshold:
                break
            last_error = error

        P = (source_pcd @ R.T) + t
        distances, indices = KNN.kneighbors(P)

        error = np.sum(distances)
        if error < best_error:
            best_error = error
            best_pose = T

    # Done, just fix if needed
    T = best_pose
    if index_targets:
        T[:3, :3] = T[:3, :3].T
        T[:3, 3] = -(T[:3, :3] @ T[:3, 3])
        T[-1, -1] = 1

    return T
