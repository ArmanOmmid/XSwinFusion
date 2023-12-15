"""Visualization utilies."""

# You can use other visualization from previous homeworks, like Open3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import itertools
import torch
import kornia.geometry.conversions as conversions

from matplotlib.pyplot import get_cmap

NUM_OBJECTS = 79
cmap = get_cmap('rainbow', NUM_OBJECTS)
COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
COLOR_PALETTE[-3] = [119, 135, 150]
COLOR_PALETTE[-2] = [176, 194, 216]
COLOR_PALETTE[-1] = [255, 255, 225]

def back_project(depth, meta, mask=None, transforms=None, samples=None, world=True):
    intrinsic = meta['intrinsic']
    R_extrinsic = meta["extrinsic"][:3, :3]
    T_extrinsic = meta["extrinsic"][:3, 3]

    if mask is None:
        v, u = np.indices(depth.shape)
    else:
        v, u = np.nonzero(mask)

    if samples is not None:
        v, u = sample_indices(v, u, samples)
        depth = depth[v, u]

    mask_indices = np.array([v, u],dtype=int)

    if transforms is not None:
        scale, translate = transforms
        if scale is not None:
            u = u / scale[0]
            v = v / scale[1]
        if translate is not None:
            u = u + translate[0]
            v = v + translate[1]

    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(depth)], axis=-1)
    points = uv1 @ np.linalg.inv(intrinsic).T * depth[..., None]  # [H, W, 3]
    if world:
        points = (points - T_extrinsic) @ R_extrinsic
    return points, mask_indices

    # Sort indices to ensure row-major order # NOTE : Doesn't seem to change anything
    # sorted_indices = np.lexsort((u, v))
    # v, u = v[sorted_indices], u[sorted_indices]

def sample_indices(v, u, max_samples):
    combined_indices = np.column_stack((v, u))
    total_indices = len(combined_indices)
    if max_samples is None or total_indices <= max_samples:
        return combined_indices.T
    
    sample_positions = np.linspace(0, total_indices - 1, num=max_samples, dtype=int)
    sample_indices = combined_indices[sample_positions]

    return sample_indices.T

def enumerate_symmetries(sym_info, inf=90):
    basis = {"x" : torch.tensor([[1, 0, 0]]), "y" : torch.tensor([[0, 1, 0]]), "z" : torch.tensor([[0, 0, 1]])}
    all_symmetries = []
    matrices = {
        "x" : [torch.eye(3).unsqueeze(0)],
        "y" : [torch.eye(3).unsqueeze(0)],
        "z": [torch.eye(3).unsqueeze(0)]
    }
    if sym_info == "no":
        return [torch.eye(3)]
    for sym_n in sym_info.split("|"):
        b = sym_n[0]
        n = int(sym_n[1]) if sym_n[1] != "i" else inf
        angles = torch.linspace(0, 2*torch.pi, steps=n+1)[1:-1] # Ignore first and last (both identity)
        for angle in angles:
            matrices[b].append(
                conversions.axis_angle_to_rotation_matrix(angle * basis[b])
            )
    for sx, sy, sz in itertools.product(*list(matrices.values())):
        all_symmetries.append(sx @ sy @ sz)

    return all_symmetries # Pad with identities (max 64)

def crop_and_resize(feature_map, mask, target_size=None, margin=12, aspect_ratio=True, mask_fill=False):

    # 144, 256
    # 288, 512
    # 432, 768
    # MUST BE GIVEN WITH H, W, C
        
    H, W = feature_map.shape[:2]

    if type(mask_fill) is int:
        feature_map[mask == False] = mask_fill

    # Identify the object's coordinates from the segmentation map
    rows, cols = np.where(mask)
    if not len(rows) or not len(cols):
        # Return the original image if no object is found in the segmentation map
        return feature_map, np.ones(3,)
    
    # Determine the bounding box
    min_row = max(rows.min() - margin, 0)
    max_row = min(rows.max() + margin, H)
    min_col = max(cols.min() - margin, 0)
    max_col = min(cols.max() + margin, W)

    if aspect_ratio:
        ratio = W / H
        # Calculate aspect ratio of the bounding box
        bbox_width = max_col - min_col
        bbox_height = max_row - min_row
        bbox_aspect_ratio = bbox_width / bbox_height

        # Adjust margins to maintain the original aspect ratio
        if bbox_aspect_ratio < ratio:
            # Increase width
            added_width = int(bbox_height * ratio) - bbox_width
            min_col = max(min_col - added_width // 2, 0)
            max_col = min(max_col + added_width // 2, W)
        elif bbox_aspect_ratio > ratio:
            # Increase height
            added_height = int(bbox_width / ratio) - bbox_height
            min_row = max(min_row - added_height // 2, 0)
            max_row = min(max_row + added_height // 2, H)

    translation = np.array([min_col, min_row, 0])  # x and y shifts, 0 for z

    # Crop the image
    feature_map = feature_map[min_row:max_row, min_col:max_col]

    if target_size is not None:
        T_H, T_W = target_size

        dtype = None
        if feature_map.dtype in [bool, np.bool8, np.bool_]:
            dtype = feature_map.dtype
            feature_map = feature_map.astype(np.uint8)

        original_crop_size = (max_col - min_col + 1, max_row - min_row + 1)
        # Its CRITICAL for depth maps for inerpolation to be INTER_NEAREST and not INTER_AREA
        feature_map = cv2.resize(feature_map, (T_W, T_H), interpolation=cv2.INTER_NEAREST)

        if dtype is not None:
            feature_map = feature_map.astype(dtype)

        scale = np.array([
            T_W / original_crop_size[0],  # x scale
            T_H / original_crop_size[1],  # y scale
            1.0])
    else:
        scale = np.ones(3,)

    return feature_map, scale, translation

def crop_and_resize_multiple(feature_maps, mask, target_size=None, margin=12, aspect_ratio=True, mask_fill=False):
    new_maps = []
    for map in feature_maps:
        new_map, s, t = crop_and_resize(map, mask, target_size=target_size, margin=margin, aspect_ratio=aspect_ratio, mask_fill=mask_fill)
        new_maps.append(new_map)
    return new_maps, s, t

def show_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([0, 4])
    ax.scatter(points[:, 0], points[:, 2], points[:, 1])

def compare_points(points1, points2, scale=1, translate=[0, 0, 0]):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xt, yt, zt = translate
    ax.set_xlim3d([-2*scale + xt, 2*scale + xt])
    ax.set_ylim3d([-2*scale + yt, 2*scale + yt])
    ax.set_zlim3d([0*scale + zt, 4*scale + zt])
    ax.scatter(points1[:, 0], points1[:, 2], points1[:, 1])
    ax.scatter(points2[:, 0], points2[:, 2], points2[:, 1])

def compare_points_triplet(source, target, prediction, truth, scale=1, translate=[0, 0, 0], figsize=(20, 20)):
    fig = plt.figure()

    points = [
        (source @ prediction[:3, :3].T + prediction[:3, 3], target),
        (source @ truth[:3, :3].T + truth[:3, 3], target),
        (source @ truth[:3, :3].T + truth[:3, 3], source @ prediction[:3, :3].T + prediction[:3, 3]),
    ]

    plt.figure(figsize=figsize)
    for i in range(3):

        points1, points2 = points[i]
        ax = plt.subplot(1, 3, i+1, projection='3d')
        xt, yt, zt = translate
        ax.set_xlim3d([-2*scale + xt, 2*scale + xt])
        ax.set_ylim3d([-2*scale + yt, 2*scale + yt])
        ax.set_zlim3d([0*scale + zt, 4*scale + zt])
        ax.scatter(points1[:, 0], points1[:, 2], points1[:, 1])
        ax.scatter(points2[:, 0], points2[:, 2], points2[:, 1])

    plt.show()

"""Metric and visualization."""
def compute_rre(R_est: np.ndarray, R_gt: np.ndarray):
    """Compute the relative rotation error (geodesic distance of rotation)."""
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)
    # relative rotation error (RRE)
    rre = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt) - 1), -1.0, 1.0))
    return rre


def compute_rte(t_est: np.ndarray, t_gt: np.ndarray):
    assert t_est.shape == (3,), 't_est: expected shape (3,), received shape {}.'.format(t_est.shape)
    assert t_gt.shape == (3,), 't_gt: expected shape (3,), received shape {}.'.format(t_gt.shape)
    # relative translation error (RTE)
    rte = np.linalg.norm(t_est - t_gt)
    return rte


VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(
    image, center, size, rotation, extrinsic, intrinsic, color=(0, 1, 0), thickness=1
):
    """Draw a projected 3D bounding box on the image.

    Args:
        image (np.ndarray): [H, W, 3] array.
        center: [3]
        size: [3]
        rotation (np.ndarray): [3, 3]
        extrinsic (np.ndarray): [4, 4]
        intrinsic (np.ndarray): [3, 3]
        color: [3]
        thickness (int): thickness of lines
    Returns:
        np.ndarray: updated image.
    """
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)

    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image
