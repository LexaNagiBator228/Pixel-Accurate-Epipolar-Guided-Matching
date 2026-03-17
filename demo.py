#!/usr/bin/env python3
"""
demo.py – Epipolar Segment Tree Matching Demo
==============================================
Generates two synthetic camera views of random 3D points, then uses
epipolar_wedge_filter_with_segment_tree to find candidate correspondences.

Compares against the brute-force epipolar_geometric_distance_filter to
verify that the segment tree achieves the same recall at a fraction of the
candidate count.

Run after building the extension:
    python demo.py
"""

import sys
from pathlib import Path

import numpy as np
import cv2

# ── import the extension ───────────────────────────────────────────────────────
try:
    from epipolar_matching import (
        epipolar_wedge_filter_with_segment_tree,
        epipolar_geometric_distance_filter,
    )
except ImportError:
    print("ERROR: epipolar_matching not found.")
    print("Build it first:  build.bat  (Windows)  |  bash build.sh  (Linux)")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Synthetic scene
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
W, H = 640, 480

# Camera intrinsics
K = np.array([[600., 0., W / 2.],
              [0., 600., H / 2.],
              [0.,   0.,     1.]], dtype=np.float32)

# Random 3D points in front of camera 1
N = 50000
pts3d = np.random.uniform(-1.5, 1.5, (N, 3)).astype(np.float32)
pts3d[:, 2] += 10.0          # push in front (Z > 0)

# Camera 1 – identity pose
R1 = np.eye(3, dtype=np.float32)
t1 = np.zeros(3, dtype=np.float32)

# Camera 2 – slight rotation + horizontal translation
rvec = np.array([0.03, 0.18, 0.01], dtype=np.float32)
R2, _ = cv2.Rodrigues(rvec)
R2 = R2.astype(np.float32)
t2 = np.array([0.5, 0.05, 0.5], dtype=np.float32)  # non-zero z avoids epipole at infinity


def project(pts: np.ndarray, K: np.ndarray,
            R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts_cam = (R @ pts.T).T + t          # (N, 3)
    pts_img = (K @ pts_cam.T).T          # (N, 3)
    return (pts_img[:, :2] / pts_img[:, 2:3]).astype(np.float32)


kp1_all = project(pts3d, K, R1, t1)
kp2_all = project(pts3d, K, R2, t2)

# Keep only points visible in both images
visible = (
    (kp1_all[:, 0] >= 0) & (kp1_all[:, 0] < W) &
    (kp1_all[:, 1] >= 0) & (kp1_all[:, 1] < H) &
    (kp2_all[:, 0] >= 0) & (kp2_all[:, 0] < W) &
    (kp2_all[:, 1] >= 0) & (kp2_all[:, 1] < H)
)
kp1 = kp1_all[visible]
kp2 = kp2_all[visible]
print(f"Visible keypoints: {len(kp1)}")


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Fundamental matrix  F = K^{-T} [t2]× R2  K^{-1}
# ══════════════════════════════════════════════════════════════════════════════
tx = np.array([[ 0,      -t2[2],  t2[1]],
               [ t2[2],  0,      -t2[0]],
               [-t2[1],  t2[0],   0    ]], dtype=np.float32)
E = tx @ R2
K_inv = np.linalg.inv(K).astype(np.float32)
F = (K_inv.T @ E @ K_inv).astype(np.float32)

# Sanity-check: for each GT pair p1↔p2, |p2^T F p1| should be ≈ 0
p1h = np.column_stack([kp1, np.ones(len(kp1), dtype=np.float32)])
p2h = np.column_stack([kp2, np.ones(len(kp2), dtype=np.float32)])
epipolar_err = np.abs(np.einsum('ij,jk,ik->i', p2h, F, p1h))
print(f"Epipolar constraint residual (mean): {epipolar_err.mean():.2e}  "
      f"(max): {epipolar_err.max():.2e}")


# kp1/kp2 are already float32 numpy arrays – pass them directly
kp1_list, kp2_list = kp1, kp2


RADIUS = 5.0        # pixel tolerance / tangent-wedge radius


# ── Brute-force baseline ───────────────────────────────────────────────────────
print(f"Running brute-force filter   (tolerance = {RADIUS} px) …")
cands_bf, dt_bf = epipolar_geometric_distance_filter(kp1_list, kp2_list, F, RADIUS)
print(f"  C++ matching time: {dt_bf:.2f} ms")


# ── Segment tree filter ────────────────────────────────────────────────────────
print(f"\nRunning segment tree filter  (radius = {RADIUS} px) …")
cands_st, dt_st = epipolar_wedge_filter_with_segment_tree(kp1_list, kp2_list, F, RADIUS)
print(f"  C++ matching time: {dt_st:.2f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Evaluate  (ground truth: kp1[i] ↔ kp2[i] by construction)
# ══════════════════════════════════════════════════════════════════════════════
n = len(kp1_list)

def evaluate(cands):
    hits      = sum(i in c for i, c in enumerate(cands))
    avg_cands = sum(map(len, cands)) / len(cands)
    return hits, avg_cands

hits_st, avg_st = evaluate(cands_st)
hits_bf, avg_bf = evaluate(cands_bf)

print(f"\n{'':─<52}")
print(f"{'Method':<20} {'Recall':>8} {'Avg cands':>11} {'Time (ms)':>10}")
print(f"{'':─<52}")
print(f"{'Segment Tree':<20} {hits_st:>5}/{n}={hits_st/n:.3f}  "
      f"{avg_st:>8.1f}    {dt_st:>8.1f}")
print(f"{'Brute Force':<20} {hits_bf:>5}/{n}={hits_bf/n:.3f}  "
      f"{avg_bf:>8.1f}    {dt_bf:>8.1f}")
print(f"{'':─<52}")
print(f"Reduction factor: {avg_bf / max(avg_st, 1):.1f}×  "
      f"(segment tree returns {avg_bf/max(avg_st,1):.1f}× fewer candidates)")


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════
img1 = np.full((H, W, 3), 245, dtype=np.uint8)
img2 = np.full((H, W, 3), 245, dtype=np.uint8)

# Draw all keypoints via vectorised numpy indexing (avoids 30k cv2.circle calls)
kp1_px = np.clip(kp1.astype(int), [0, 0], [W - 1, H - 1])
kp2_px = np.clip(kp2.astype(int), [0, 0], [W - 1, H - 1])
img1[kp1_px[:, 1], kp1_px[:, 0]] = (150, 150, 150)
img2[kp2_px[:, 1], kp2_px[:, 0]] = (150, 150, 150)

vis = np.hstack([img1, img2])

# Draw matched pairs (segment tree hits only)
rng = np.random.default_rng(0)
draw_idx = rng.choice(n, size=min(50, n), replace=False)
for i in draw_idx:
    if i not in cands_st[i]:
        continue
    color = tuple(rng.integers(40, 210, 3).tolist())
    p1 = (int(kp1[i, 0]),     int(kp1[i, 1]))
    p2 = (int(kp2[i, 0]) + W, int(kp2[i, 1]))
    cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)
    cv2.circle(vis, p1, 4, color, -1)
    cv2.circle(vis, p2, 4, color, -1)


# Labels
font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
cv2.putText(vis, "Image 1",
            (10, H - 12), font, fs, (80, 80, 80), thick)
cv2.putText(vis, "Image 2",
            (W + 10, H - 12), font, fs, (80, 80, 80), thick)
cv2.putText(vis,
            f"Segment tree  |  recall={hits_st/n:.3f}  avg_cands={avg_st:.1f}  "
            f"({dt_st:.1f} ms)",
            (10, 22), font, fs, (30, 30, 30), thick)

out_path = Path(__file__).parent / "demo_matches.png"
cv2.imwrite(str(out_path), vis)
print(f"\nVisualization saved to  {out_path}")

try:
    cv2.imshow("Epipolar Segment Tree Matching Demo", vis)
    print("Press any key to close …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass   # headless environment
