#!/usr/bin/env python3
"""
demo_real.py – Guided SIFT Matching on Real Calibrated Images
=============================================================
Loads two fisheye images from ./exc/, undistorts them, extracts SIFT
descriptors, then matches them using epipolar-guided candidate filtering
followed by nearest-neighbour + ratio test among the candidates only.

Compares all three epipolar filters: segment-tree (optimised), segment-tree (origin), and angular hash.

Run after building the extension:
    python demo_real.py
"""

import sys
from pathlib import Path

import numpy as np
import cv2

try:
    from epipolar_matching import (
        epipolar_wedge_filter_with_segment_tree,
        epipolar_hash_filter_cpp,
    )
except ImportError:
    print("ERROR: epipolar_matching not found.")
    print("Build it first:  bash build.sh")
    sys.exit(1)


EXC_DIR = Path(__file__).parent / "exc"
SCALE   = 0.5   # downsample factor 
RADIUS  = 5.0    # epipolar tolerance / wedge radius in scaled pixels
RATIO   = 0.8    # Lowe ratio-test threshold


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Parse COLMAP cameras.txt and images_mod.txt
# ══════════════════════════════════════════════════════════════════════════════

def parse_cameras(path):
    """THIN_PRISM_FISHEYE → {cam_id: (K 3×3 float64, D [k1,k2,k3,k4] float64)}"""
    cams = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        cid = int(parts[0])
        # params: fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
        p = [float(x) for x in parts[4:]]
        K = np.array([[p[0], 0, p[2]],
                      [0, p[1], p[3]],
                      [0,    0,    1]], dtype=np.float64)
        D = np.array([p[4], p[5], p[8], p[9]], dtype=np.float64)  # k1 k2 k3 k4
        cams[cid] = (K, D)
    return cams


def quat_to_R(qw, qx, qy, qz):
    """COLMAP quaternion (qw first) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def parse_images(path):
    """Returns {name_stem: (R 3×3, t 3-vec, cam_id)}"""
    imgs = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        R = quat_to_R(*map(float, parts[1:5]))
        t = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cid = int(parts[8])
        stem = Path(parts[9]).stem
        imgs[stem] = (R, t, cid)
    return imgs


cameras   = parse_cameras(EXC_DIR / "cameras.txt")
img_poses = parse_images(EXC_DIR / "images_mod.txt")

R1, t1, cid1 = img_poses["DSC_0320"]
R2, t2, cid2 = img_poses["DSC_0321"]
K1_orig, D1  = cameras[cid1]
K2_orig, D2  = cameras[cid2]


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Load, undistort (fisheye), and scale images
# ══════════════════════════════════════════════════════════════════════════════

def load_undistort(img_path, K, D, scale):
    img = cv2.imread(str(img_path))
    assert img is not None, f"Cannot load {img_path}"
    h, w = img.shape[:2]
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    small  = cv2.resize(undist, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    K_scaled = K_new * scale
    K_scaled[2, 2] = 1.0
    return small, K_scaled


print("Loading and undistorting images …")
img1, K1 = load_undistort(EXC_DIR / "DSC_0320.JPG", K1_orig, D1, SCALE)
img2, K2 = load_undistort(EXC_DIR / "DSC_0323.JPG", K2_orig, D2, SCALE)
print(f"  Scaled size: {img1.shape[1]}×{img1.shape[0]}")


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Fundamental matrix from calibrated poses
#      COLMAP convention: P_cam = R * P_world + t
# ══════════════════════════════════════════════════════════════════════════════

R_rel = R2 @ R1.T
t_rel = t2 - R_rel @ t1

def skew(v):
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=np.float64)

E = skew(t_rel) @ R_rel
F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
F = (F / np.abs(F).max()).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  SIFT extraction
# ══════════════════════════════════════════════════════════════════════════════

print("Extracting SIFT features …")
sift = cv2.SIFT_create(nfeatures=50000, )
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kpts1, desc1 = sift.detectAndCompute(g1, None)
kpts2, desc2 = sift.detectAndCompute(g2, None)
print(f"  Detected keypoints: {len(kpts1)} in img1, {len(kpts2)} in img2")
print(f"  kp1={len(kpts1)},  kp2={len(kpts2)}")

kp1_arr = np.array([[k.pt[0], k.pt[1]] for k in kpts1], dtype=np.float32)
kp2_arr = np.array([[k.pt[0], k.pt[1]] for k in kpts2], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Epipolar-guided candidate filtering
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nRunning segment-tree wedge filter  (radius={RADIUS} px) …")
cands_st, dt_st = epipolar_wedge_filter_with_segment_tree(kp1_arr, kp2_arr, F, RADIUS)
print(f"  C++ filter time: {dt_st:.2f} ms")

print(f"Running angular hash filter        (tol={RADIUS} px, bins=45) …")
cands_hash, dt_hash = epipolar_hash_filter_cpp(kp1_arr, kp2_arr, F, tol=RADIUS)
print(f"  C++ filter time: {dt_hash:.2f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  6.  Descriptor matching within candidates (ratio test)
# ══════════════════════════════════════════════════════════════════════════════

def match_within_candidates(desc1, desc2, candidates, ratio=RATIO):
    """Nearest-neighbour + Lowe ratio test restricted to candidate set."""
    matches = []
    for i, cands in enumerate(candidates):
        if len(cands) < 2:
            continue
        d1 = desc1[i].astype(np.float32)
        d2 = desc2[np.array(cands)].astype(np.float32)
        dists = np.sum((d2 - d1) ** 2, axis=1)
        order = np.argsort(dists)
        if dists[order[0]] < ratio ** 2 * dists[order[1]]:
            matches.append((i, cands[order[0]], float(np.sqrt(dists[order[0]]))))
    return matches


print("\nDescriptor matching within candidates …")
matches_st   = match_within_candidates(desc1, desc2, cands_st)
matches_hash = match_within_candidates(desc1, desc2, cands_hash)

# ── OpenCV BF matcher without any epipolar constraint ─────────────────────────
import time as _time
print("Running OpenCV BF matcher (no epipolar, ratio=0.8) …")
_bfm = cv2.BFMatcher(cv2.NORM_L2)
_t0 = _time.perf_counter()
_raw = _bfm.knnMatch(desc1, desc2, k=2)
_t1 = _time.perf_counter()
dt_cv_bf = (_t1 - _t0) * 1000
matches_cv_bf = [(m.queryIdx, m.trainIdx, m.distance)
                 for m, n in _raw if m.distance < 0.8 * n.distance]

avg_st   = sum(len(c) for c in cands_st)   / max(len(cands_st),   1)
avg_hash = sum(len(c) for c in cands_hash) / max(len(cands_hash), 1)

print(f"\n{'':─<72}")
print(f"{'Method':<30} {'Matches':>9} {'Avg cands':>11} {'Time (ms)':>10}")
print(f"{'':─<72}")
print(f"{'CV BF (no epipolar)':<30} {len(matches_cv_bf):>9}  {'N/A':>9}  {dt_cv_bf:>9.1f}")
print(f"{'Seg Tree (optimised)':<30} {len(matches_st):>9}  {avg_st:>9.1f}  {dt_st:>9.1f}")
print(f"{'Angular Hash':<30} {len(matches_hash):>9}  {avg_hash:>9.1f}  {dt_hash:>9.1f}")
print(f"{'':─<72}")


# ══════════════════════════════════════════════════════════════════════════════
#  7.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

vis = np.hstack([img1, img2])
W = img1.shape[1]

rng = np.random.default_rng(1)
draw_idx = rng.choice(len(matches_st), size=min(300, len(matches_st)), replace=False)
for idx in draw_idx:
    i, j, _ = matches_st[idx]
    p1 = (int(kpts1[i].pt[0]),     int(kpts1[i].pt[1]))
    p2 = (int(kpts2[j].pt[0]) + W, int(kpts2[j].pt[1]))
    color = tuple(rng.integers(40, 210, 3).tolist())
    cv2.line(vis,   p1, p2, color, 1, cv2.LINE_AA)
    cv2.circle(vis, p1, 3,  color, -1)
    cv2.circle(vis, p2, 3,  color, -1)

font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
H = img1.shape[0]
cv2.putText(vis, "DSC_0320 (undistorted)", (10, H - 12), font, fs, (80, 80, 80), th)
cv2.putText(vis, "DSC_0323 (undistorted)", (W + 10, H - 12), font, fs, (80, 80, 80), th)
cv2.putText(vis,
    f"CV-BF: {len(matches_cv_bf)}m {dt_cv_bf:.0f}ms  |  "
    f"ST-optim: {len(matches_st)}m {dt_st:.0f}ms  |  "
    f"Hash: {len(matches_hash)}m {dt_hash:.0f}ms",
    (10, 22), font, fs, (20, 20, 200), th)

out_path = Path(__file__).parent / "demo_real_matches.png"
cv2.imwrite(str(out_path), vis)
print(f"\nVisualization saved to {out_path}")

try:
    cv2.imshow("Guided SIFT Matching – Real Images", vis)
    print("Press any key to close …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass
