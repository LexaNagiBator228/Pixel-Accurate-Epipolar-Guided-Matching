#!/usr/bin/env python3
"""
demo.py – Epipolar Segment Tree Matching Demo
==============================================
Generates two synthetic camera views of random 3D points, then uses
epipolar_wedge_filter_with_segment_tree to find candidate correspondences.

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
        epipolar_wedge_filter_with_segment_tree_origin,
        epipolar_hash_filter_cpp,
    )
except ImportError:
    print("ERROR: epipolar_matching not found.")
    print("Build it first:  build.bat  (Windows)  |  bash build.sh  (Linux)")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Synthetic scene
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
W, H = 640*2, 480*2

# Camera intrinsics
K = np.array([[600., 0., W / 2.],
              [0., 600., H / 2.],
              [0.,   0.,     1.]], dtype=np.float32)

# Random 3D points in front of camera 1
N = 10000
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


# ══════════════════════════════════════════════════════════════════════════════
#  3a. Synthetic SIFT descriptors for descriptor matching baseline
#      (128-dim random unit vectors, one per keypoint)
# ══════════════════════════════════════════════════════════════════════════════
rng_desc = np.random.default_rng(0)
desc_dim = 128
# Assign each visible point a random descriptor; GT pairs share the same descriptor
# so BF matcher can find them.
raw = rng_desc.standard_normal((N, desc_dim)).astype(np.float32)
raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-9
vis_idx = np.where(visible)[0]
desc1 = raw[vis_idx]   # desc for kp1
desc2 = raw[vis_idx]   # same descriptors → GT pairs have distance 0
# Add noise to make it realistic
desc2 = desc2 + rng_desc.standard_normal(desc2.shape).astype(np.float32) * 0.1
desc2 /= np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-9

import time as _time

# ── OpenCV BF descriptor matcher (no epipolar, Lowe ratio 0.8) ────────────────
print("\nRunning OpenCV BF descriptor matcher (no epipolar geometry) …")
bf = cv2.BFMatcher(cv2.NORM_L2)
_t0 = _time.perf_counter()
raw_matches = bf.knnMatch(desc1, desc2, k=2)
_t1 = _time.perf_counter()
dt_cv_bf = (_t1 - _t0) * 1000
good_cv_bf = [m for m, n in raw_matches if m.distance < 0.8 * n.distance]
print(f"  OpenCV BF time: {dt_cv_bf:.2f} ms   matches: {len(good_cv_bf)}")

# ── Segment tree filter (origin) ───────────────────────────────────────────────
print(f"\nRunning segment tree filter (origin) (radius = {RADIUS} px) …")
cands_or, dt_or = epipolar_wedge_filter_with_segment_tree_origin(kp1_list, kp2_list, F, RADIUS)
print(f"  C++ matching time: {dt_or:.2f} ms")

# ── Segment tree filter (optimised) ───────────────────────────────────────────
print(f"\nRunning segment tree filter (optim)  (radius = {RADIUS} px) …")
cands_st, dt_st = epipolar_wedge_filter_with_segment_tree(kp1_list, kp2_list, F, RADIUS)
print(f"  C++ matching time: {dt_st:.2f} ms")

# ── Angular hash filter ────────────────────────────────────────────────────────
print(f"\nRunning angular hash filter         (tolerance = {RADIUS} px) …")
cands_hash, dt_hash = epipolar_hash_filter_cpp(kp1_list, kp2_list, F, tol=RADIUS)
print(f"  C++ matching time: {dt_hash:.2f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Evaluate  (ground truth: kp1[i] ↔ kp2[i] by construction)
# ══════════════════════════════════════════════════════════════════════════════
n = len(kp1_list)

def evaluate(cands):
    hits      = sum(i in c for i, c in enumerate(cands))
    avg_cands = sum(map(len, cands)) / len(cands)
    return hits, avg_cands

hits_or,   avg_or   = evaluate(cands_or)
hits_st,   avg_st   = evaluate(cands_st)
hits_hash, avg_hash = evaluate(cands_hash)

print(f"\n{'':─<72}")
print(f"{'Method':<30} {'Recall':>8} {'Avg cands':>11} {'Time (ms)':>10}")
print(f"{'':─<72}")
print(f"{'CV BF (no epipolar)':<30} {len(good_cv_bf)/n:.3f}  {'N/A':>9}  {dt_cv_bf:>9.1f}")
print(f"{'Seg Tree (origin)':<30} {hits_or/n:.3f}  {avg_or:>9.1f}  {dt_or:>9.1f}")
print(f"{'Seg Tree (optimised)':<30} {hits_st/n:.3f}  {avg_st:>9.1f}  {dt_st:>9.1f}")
print(f"{'Angular Hash':<30} {hits_hash/n:.3f}  {avg_hash:>9.1f}  {dt_hash:>9.1f}")
print(f"{'':─<72}")


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
            f"ST-origin {dt_or:.1f}ms  |  ST-optim {dt_st:.1f}ms  |  "
            f"Hash {dt_hash:.1f}ms",
            (10, 22), font, fs * 0.75, (30, 30, 30), thick)

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
