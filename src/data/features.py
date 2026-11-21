"""Shared feature engineering layer.

Computes the canonical 24 base features, 7 edge features, 6 global swarm features,
and 30 causal lag differences. Single source of truth for all four models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.config import (
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    EDGE_FEATURES_7,
    GLOBAL_FEATURES_6,
    LAG_BASE_MAP,
    LAG_OFFSETS,
    LAG_DIFF_FEATURES_30,
    LGBM_FEATURES_52,
    LGBM_FEATURES_54,
    FORBIDDEN_COLUMNS,
    SENSING_RADIUS,
)


def compute_derived_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived geometric and graph features on a raw sorted DataFrame."""
    df = df.copy()

    # Basic trigonometric and kinematic alignment features
    df["heading_sin"] = np.sin(df["heading"].to_numpy(dtype=np.float32))
    df["heading_cos"] = np.cos(df["heading"].to_numpy(dtype=np.float32))

    # Velocity alignment with swarm centroid velocity (replaces redundant neighbor_frac)
    vc = df.groupby(["case_id", "step"], observed=True)[["vx", "vy", "vz"]].transform("mean").to_numpy(dtype=np.float32)
    vel = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float32)
    v_norm = np.linalg.norm(vel, axis=1) + 1e-6
    vc_norm = np.linalg.norm(vc, axis=1) + 1e-6
    df["centroid_vel_alignment"] = (np.sum(vel * vc, axis=1) / (v_norm * vc_norm)).astype(np.float32)

    df["is_isolated"] = (df["num_neighbors"] == 0).astype(np.float32)

    # Pre-allocate arrays for snapshot-level derived metrics
    n_rows = len(df)
    deg_c = np.zeros(n_rows, dtype=np.float32)
    r15 = np.zeros(n_rows, dtype=np.float32)
    r30 = np.zeros(n_rows, dtype=np.float32)
    r45 = np.zeros(n_rows, dtype=np.float32)
    contrast = np.zeros(n_rows, dtype=np.float32)
    entropies = np.zeros(n_rows, dtype=np.float32)

    # Sort strictly by (case_id, step, agent_id) for snapshot iteration
    df_sorted = df.sort_values(["case_id", "step", "agent_id"]).reset_index()
    orig_indices = df_sorted["index"].to_numpy()

    # Process each snapshot
    ln12 = float(np.log(12.0))
    groups = df_sorted.groupby(["case_id", "step"], observed=True)

    pos_all = df_sorted[["rel_x", "rel_y", "rel_z"]].to_numpy(dtype=np.float32)
    head_all = df_sorted["heading"].to_numpy(dtype=np.float32)
    dens_all = df_sorted["weighted_density"].to_numpy(dtype=np.float32)

    for _, grp_idx in groups.indices.items():
        n = len(grp_idx)
        if n <= 1:
            continue
        pts = pos_all[grp_idx]
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)

        m55 = D <= SENSING_RADIUS
        counts = m55.sum(axis=1)

        deg_c[grp_idx] = counts / float(max(n - 1, 1))
        r15[grp_idx] = (D <= 15.0).sum(axis=1)
        r30[grp_idx] = (D <= 30.0).sum(axis=1)
        r45[grp_idx] = (D <= 45.0).sum(axis=1)

        # Local density contrast: own density - mean neighbor density
        dens = dens_all[grp_idx]
        neighbor_sums = m55.astype(np.float32) @ dens
        mean_nb_dens = np.where(counts > 0, neighbor_sums / np.maximum(counts, 1), 0.0)
        contrast[grp_idx] = np.where(counts > 0, dens - mean_nb_dens, 0.0)

        # Nearest neighbor bearing entropy (12 azimuth sectors in body heading frame)
        head = head_all[grp_idx]
        dx = pts[:, 0][None, :] - pts[:, 0][:, None]
        dy = pts[:, 1][None, :] - pts[:, 1][:, None]
        bearing = np.arctan2(dy, dx) - head[:, None]
        bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi
        bins = np.floor((bearing + np.pi) / (2.0 * np.pi) * 12.0).astype(int) % 12

        ent_snap = np.zeros(n, dtype=np.float32)
        valid_agents = np.where(counts > 0)[0]
        for i in valid_agents:
            c = counts[i]
            row_bins = bins[i][m55[i]]
            b_counts = np.bincount(row_bins, minlength=12)
            p = b_counts[b_counts > 0] / float(c)
            ent_snap[i] = -float(np.sum(p * np.log(p))) / ln12
        entropies[grp_idx] = ent_snap

    # Map back to original row order
    df.loc[orig_indices, "degree_centrality_c"] = deg_c
    df.loc[orig_indices, "n_within_r15"] = r15
    df.loc[orig_indices, "n_within_r30"] = r30
    df.loc[orig_indices, "n_within_r45"] = r45
    df.loc[orig_indices, "local_density_contrast"] = contrast
    df.loc[orig_indices, "nn_bearing_entropy"] = entropies

    return df


def compute_causal_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal within-agent lag differences over offsets 1, 2, 4, 8, 16."""
    df = df.sort_values(["case_id", "agent_id", "step"]).copy()
    g = df.groupby(["case_id", "agent_id"], observed=True)

    for tag, src_col in LAG_BASE_MAP.items():
        for lag in LAG_OFFSETS:
            df[f"d{lag}_{tag}"] = g[src_col].diff(lag)

    return df


def extract_snapshot_graph(
    sub_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract node features, edge_index, edge_attr, and global features for one snapshot.

    Returns:
        x: [N, 22] float32 node features in canonical order
        edge_index: [2, E] int64 directed edge connectivity with self-loops
        edge_attr: [E, 7] float32 edge attributes in receiver frame
        global_feat: [6] float32 swarm-level context vector
    """
    n = len(sub_df)
    x = sub_df[BASE_FEATURES_22].to_numpy(dtype=np.float32)
    pts = sub_df[["rel_x", "rel_y", "rel_z"]].to_numpy(dtype=np.float32)
    vel = sub_df[["vx", "vy", "vz"]].to_numpy(dtype=np.float32)
    head = sub_df["heading"].to_numpy(dtype=np.float32)

    # 1. 3D Euclidean distances
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)

    # 2. Directed edges (src -> dst) within sensing radius 55m
    ii, jj = np.where(D <= SENSING_RADIUS)
    if len(ii) == 0:
        # Fallback for isolated swarm: self-edges
        ii = np.arange(n, dtype=np.int64)
        jj = np.arange(n, dtype=np.int64)
        edge_index = np.stack([ii, jj], axis=0)
        edge_attr = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            (n, 1),
        )
    else:
        edge_index = np.stack([ii, jj], axis=0).astype(np.int64)
        rel = (pts[jj] - pts[ii]).astype(np.float32)
        dist = np.linalg.norm(rel, axis=1) + 1e-6
        unit = rel / dist[:, None]
        rv = (vel[jj] - vel[ii]).astype(np.float32)
        rel_speed = np.linalg.norm(rv, axis=1).astype(np.float32)
        closing = np.sum(rv * unit, axis=1).astype(np.float32)
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - head[ii]
        dh = head[jj] - head[ii]
        edge_attr = np.stack(
            [
                (dist / SENSING_RADIUS).astype(np.float32),
                rel_speed,
                closing,
                np.cos(ang).astype(np.float32),
                np.sin(ang).astype(np.float32),
                np.cos(dh).astype(np.float32),
                np.sin(dh).astype(np.float32),
            ],
            axis=1,
        )

    # 4. Global swarm context vector (6 features)
    alive_count = float(n)
    swarm_spread = float(np.sqrt(np.mean(sub_df["dist_to_centroid"].to_numpy() ** 2)))
    swarm_centroid_speed = float(np.linalg.norm(np.mean(vel, axis=0)))

    n_undir = len(ii) // 2
    density = 2.0 * n_undir / float(n * (n - 1)) if n >= 2 else 0.0

    if len(ii) > 0:
        adj = csr_matrix((np.ones(len(ii), dtype=int), (ii, jj)), shape=(n, n))
        n_comp, labels = connected_components(adj, directed=False)
        largest_frac = float(np.bincount(labels).max()) / float(n)
    else:
        n_comp = n
        largest_frac = 1.0 / float(n)

    global_feat = np.array(
        [alive_count, swarm_spread, swarm_centroid_speed, density, float(n_comp), largest_frac],
        dtype=np.float32,
    )

    return x, edge_index, edge_attr, global_feat


def validate_feature_matrix(feature_names: list[str]) -> None:
    """Enforce strict forbidden column checks before feeding features to any model."""
    forbidden_present = set(feature_names) & FORBIDDEN_COLUMNS
    if forbidden_present:
        raise ValueError(
            f"LEAKAGE SECURITY VIOLATION: Forbidden columns present in model input: {forbidden_present}"
        )


def derive_features_from_telemetry(
    snapshot_df: pd.DataFrame,
    prev_snapshot_df: pd.DataFrame | None = None,
    dt_obs: float = 0.10,
) -> pd.DataFrame:
    """Derive all 22 canonical base features strictly from raw position and velocity telemetry.

    Requires ONLY:
        - agent_id: int
        - 3D Position: (x, y, z) or (px, py, pz) or (rel_x, rel_y, rel_z)
        - 3D Velocity: (vx, vy, vz)
    Optional:
        - step (default: 0)
        - case_id (default: 'live')
        - prev_snapshot_df: for causal finite differences of accel and heading_rate
    """
    df = snapshot_df.copy()

    # 1. Standardize column names
    col_map = {}
    for c in ["px", "Px", "pos_x", "posX"]:
        if c in df.columns:
            col_map[c] = "x"
    for c in ["py", "Py", "pos_y", "posY"]:
        if c in df.columns:
            col_map[c] = "y"
    for c in ["pz", "Pz", "pos_z", "posZ"]:
        if c in df.columns:
            col_map[c] = "z"
    for c in ["Vx", "vel_x", "velX"]:
        if c in df.columns:
            col_map[c] = "vx"
    for c in ["Vy", "vel_y", "velY"]:
        if c in df.columns:
            col_map[c] = "vy"
    for c in ["Vz", "vel_z", "velZ"]:
        if c in df.columns:
            col_map[c] = "vz"
    if col_map:
        df = df.rename(columns=col_map)

    if "x" not in df.columns and "rel_x" in df.columns:
        df["x"] = df["rel_x"]
        df["y"] = df["rel_y"]
        df["z"] = df["rel_z"]

    if "case_id" not in df.columns:
        df["case_id"] = "live"
    if "step" not in df.columns:
        df["step"] = 0

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=BASE_FEATURES_22 + ["agent_id", "step", "case_id"])

    # 2. Position & Centroid
    pos = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    c_pos = np.mean(pos, axis=0, keepdims=True)
    rel_pos = pos - c_pos
    df["rel_x"] = rel_pos[:, 0]
    df["rel_y"] = rel_pos[:, 1]
    df["rel_z"] = rel_pos[:, 2]

    dist_to_centroid = np.linalg.norm(rel_pos, axis=1)
    df["dist_to_centroid"] = dist_to_centroid.astype(np.float32)

    # Rank dist to centroid in [0, 1]
    if n > 1:
        ranks = pd.Series(dist_to_centroid).rank(method="min").to_numpy(dtype=np.float32)
        df["rank_dist_to_centroid"] = (ranks - 1.0) / float(n - 1)
    else:
        df["rank_dist_to_centroid"] = 0.0

    # 3. Kinematics from Velocity
    vel = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float32)
    speed = np.linalg.norm(vel, axis=1).astype(np.float32)
    df["speed"] = speed

    h_spd = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2).astype(np.float32)
    h_spd_safe = np.maximum(h_spd, 1e-6)

    heading = np.arctan2(vel[:, 1], vel[:, 0]).astype(np.float32)
    df["heading"] = heading
    df["heading_sin"] = (vel[:, 1] / h_spd_safe).astype(np.float32)
    df["heading_cos"] = (vel[:, 0] / h_spd_safe).astype(np.float32)
    df["flight_path_angle"] = np.arctan2(vel[:, 2], h_spd_safe).astype(np.float32)

    # Centroid velocity alignment
    v_c = np.mean(vel, axis=0, keepdims=True)
    v_c_norm = np.linalg.norm(v_c) + 1e-6
    v_norm = speed + 1e-6
    df["centroid_vel_alignment"] = (np.sum(vel * v_c, axis=1) / (v_norm * v_c_norm)).astype(np.float32)

    # 4. Dynamic finite differences (accel & heading_rate)
    df["accel"] = 0.0
    df["heading_rate"] = 0.0

    if prev_snapshot_df is not None and len(prev_snapshot_df) > 0:
        prev = prev_snapshot_df.copy()
        # Compute prev speed and heading if not present
        if "vx" in prev.columns and "vy" in prev.columns and "vz" in prev.columns:
            prev_vel = prev[["vx", "vy", "vz"]].to_numpy(dtype=np.float32)
            prev_speed = np.linalg.norm(prev_vel, axis=1).astype(np.float32)
            prev_heading = np.arctan2(prev_vel[:, 1], prev_vel[:, 0]).astype(np.float32)
            prev_speed_map = dict(zip(prev["agent_id"], prev_speed))
            prev_heading_map = dict(zip(prev["agent_id"], prev_heading))

            for idx, row in df.iterrows():
                aid = row["agent_id"]
                if aid in prev_speed_map:
                    sp_curr = row["speed"]
                    sp_prev = prev_speed_map[aid]
                    df.at[idx, "accel"] = (sp_curr - sp_prev) / float(dt_obs)

                    h_curr = row["heading"]
                    h_prev = prev_heading_map[aid]
                    dh = (h_curr - h_prev + np.pi) % (2.0 * np.pi) - np.pi
                    df.at[idx, "heading_rate"] = dh / float(dt_obs)

    # 5. Spatial Escort Geometry from Pairwise Distances
    D = np.linalg.norm(rel_pos[:, None, :] - rel_pos[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)

    m55 = D <= SENSING_RADIUS
    num_nb = m55.sum(axis=1)
    df["num_neighbors"] = num_nb.astype(np.float32)
    df["is_isolated"] = (num_nb == 0).astype(np.float32)

    df["degree_centrality_c"] = (num_nb / float(max(n - 1, 1))).astype(np.float32)
    df["n_within_r15"] = (D <= 15.0).sum(axis=1).astype(np.float32)
    df["n_within_r30"] = (D <= 30.0).sum(axis=1).astype(np.float32)
    df["n_within_r45"] = (D <= 45.0).sum(axis=1).astype(np.float32)

    # Min neighbor distance
    df["min_neighbor_dist"] = np.min(D, axis=1).astype(np.float32)

    # Mean neighbor distance (within sensing radius)
    mean_nb = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if num_nb[i] > 0:
            mean_nb[i] = np.mean(D[i, m55[i]])
        else:
            mean_nb[i] = SENSING_RADIUS
    df["mean_neighbor_dist"] = mean_nb

    # Weighted density: sum(exp(-dist^2 / 324.0)) for neighbors <= 55m
    w_dens = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if num_nb[i] > 0:
            w_dens[i] = np.sum(np.exp(-D[i, m55[i]] ** 2 / 324.0))
    df["weighted_density"] = w_dens

    # Local density contrast: own density - mean neighbor density
    contrast = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if num_nb[i] > 0:
            mean_nb_dens = np.mean(w_dens[m55[i]])
            contrast[i] = w_dens[i] - mean_nb_dens
    df["local_density_contrast"] = contrast

    # Angular coverage: 48 sectors (12 azimuth x 4 elevation)
    ang_cov = np.zeros(n, dtype=np.float32)
    entropies = np.zeros(n, dtype=np.float32)
    ln12 = float(np.log(12.0))

    for i in range(n):
        if num_nb[i] == 0:
            continue
        diff = rel_pos - rel_pos[i]
        d_sub = D[i]
        valid = d_sub <= SENSING_RADIUS

        dx = diff[valid, 0]
        dy = diff[valid, 1]
        dz = diff[valid, 2]
        d_xy = np.sqrt(dx ** 2 + dy ** 2)

        # 48 sectors: 12 azimuth x 4 elevation
        az = np.arctan2(dy, dx)
        az_bin = np.floor((az + np.pi) / (2.0 * np.pi) * 12.0).astype(int).clip(0, 11)
        el = np.arctan2(dz, np.maximum(d_xy, 1e-6))
        el_bin = np.floor((el + np.pi / 2.0) / np.pi * 4.0).astype(int).clip(0, 3)
        sector_idx = az_bin * 4 + el_bin
        ang_cov[i] = len(np.unique(sector_idx)) / 48.0

        # Nearest neighbor bearing entropy (relative to agent body heading)
        head_i = heading[i]
        bearing = (az - head_i + np.pi) % (2.0 * np.pi) - np.pi
        b_bins = np.floor((bearing + np.pi) / (2.0 * np.pi) * 12.0).astype(int).clip(0, 11)
        b_counts = np.bincount(b_bins, minlength=12)
        p = b_counts[b_counts > 0] / float(num_nb[i])
        entropies[i] = -float(np.sum(p * np.log(p))) / ln12

    df["angular_coverage"] = ang_cov
    df["nn_bearing_entropy"] = entropies

    return df


