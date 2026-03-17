// guided_desc_match.cpp
// Pixel-accurate epipolar guided matching via segment tree
// Exposes two functions to Python:
//   epipolar_wedge_filter_with_segment_tree  – O(log N + K) per query
//   epipolar_geometric_distance_filter        – O(N*M) brute-force baseline

// MSVC does not define M_PI by default
#define _USE_MATH_DEFINES

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#ifdef __AVX512F__
#  include <immintrin.h>
#endif

namespace py = pybind11;

/* ── SoA point storage ────────────────────────────────────────────────────── */
// Separate contiguous x[] and y[] arrays let the compiler auto-vectorise
// the brute-force inner loop and avoid std::get<> indirection.
struct Points {
    std::vector<float> x, y;
    std::size_t size() const noexcept { return x.size(); }
};

using CandidateList = std::vector<std::vector<int>>;

/* ── angle helper ─────────────────────────────────────────────────────────── */
// Conditional arithmetic is far faster than fmod for angles already
// in [-max_angle, 2*max_angle).
inline float wrap_angle(float a, float max_angle) noexcept
{
    if (a >= max_angle) a -= max_angle;
    if (a < 0.f)        a += max_angle;
    return a;
}

/* ── numpy (N,2) float32 → SoA ────────────────────────────────────────────── */
static Points from_numpy(const py::array_t<float> &arr)
{
    const auto r = arr.unchecked<2>();
    const py::ssize_t n = arr.shape(0);
    Points p;
    p.x.resize(n);
    p.y.resize(n);
    for (py::ssize_t i = 0; i < n; ++i) {
        p.x[i] = r(i, 0);
        p.y[i] = r(i, 1);
    }
    return p;
}

/* ── epipole via Eigen SVD (stack-allocated, no OpenCV) ───────────────────── */
// Returns the right null-vector of M, normalised so that e(2) == 1.
static Eigen::Vector3f computeEpipole(const Eigen::Matrix3f &M)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullV);
    Eigen::Vector3f e = svd.matrixV().col(2);
    e /= e(2);
    return e;
}

/* ── tangent-wedge angular interval ──────────────────────────────────────── */
// Returns the angular interval [a1, a2] (degrees) subtended by the disk of
// radius R centred at keypoint (kx,ky), as seen from epipole (ex,ey).
// Returns false when the epipole is inside the disk.
inline bool compute_tangent_angles(float ex, float ey,
                                   float kx, float ky, float R,
                                   float &a1, float &a2,
                                   float max_angle) noexcept
{
    constexpr float kRad2Deg = 180.f / float(M_PI);
    const float dx = ex - kx;
    const float dy = ey - ky;
    const float d2 = dx * dx + dy * dy;
    if (d2 <= R * R) return false;
    const float inv_d2 = 1.f / d2;
    const float l  = R * R * inv_d2;
    const float m  = R * std::sqrt(d2 - R * R) * inv_d2;
    const float vx = l * dx,  vy = l * dy;
    const float qx = -m * dy, qy = m * dx;
    a1 = wrap_angle(std::atan2(ky + vy + qy - ey, kx + vx + qx - ex) * kRad2Deg, max_angle);
    a2 = wrap_angle(std::atan2(ky + vy - qy - ey, kx + vx - qx - ex) * kRad2Deg, max_angle);
    if (wrap_angle(a2 - a1, max_angle) > 180.f) std::swap(a1, a2);
    return true;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  BRUTE-FORCE EPIPOLAR DISTANCE FILTER
 *  For each kp1[i] keep every kp2[j] whose distance to the epipolar line ≤ tol
 * ════════════════════════════════════════════════════════════════════════════ */
static CandidateList epipolar_geometric_distance_filter(
    const Points &kp1, const Points &kp2,
    const Eigen::Ref<const Eigen::Matrix3f> &F,
    float tolerance_px)
{
    const int N1 = static_cast<int>(kp1.size());
    const int N2 = static_cast<int>(kp2.size());

    CandidateList cand(N1);
    for (auto &v : cand) v.reserve(8);

    // Raw pointers let the compiler prove non-aliasing and emit SIMD code.
    const float *__restrict kp2x = kp2.x.data();
    const float *__restrict kp2y = kp2.y.data();

    // Work is uniform across iterations → schedule(static) has lower overhead
    // than schedule(dynamic).
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N1; ++i)
    {
        const float p1x = kp1.x[i], p1y = kp1.y[i];
        // l = F * [p1x, p1y, 1]^T
        const float a = F(0,0)*p1x + F(0,1)*p1y + F(0,2);
        const float b = F(1,0)*p1x + F(1,1)*p1y + F(1,2);
        const float c = F(2,0)*p1x + F(2,1)*p1y + F(2,2);
        // Distance check without sqrt or division:
        //   |a*x + b*y + c| / sqrt(a²+b²) <= tol
        //   ↔  (a*x + b*y + c)² <= tol² * (a² + b²)
        // The inner loop is now purely FMA + compare → auto-vectorisable.
        const float tol_sq_norm = tolerance_px * tolerance_px * (a*a + b*b);
        for (int j = 0; j < N2; ++j)
        {
            const float num = a * kp2x[j] + b * kp2y[j] + c;
            if (num * num <= tol_sq_norm)
                cand[i].push_back(j);
        }
    }
    return cand;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SEGMENT-TREE EPIPOLAR WEDGE FILTER  (original / separate-vectors variant)
 *
 *  Same algorithm as epipolar_wedge_filter_with_segment_tree but stores the
 *  tree in separate parallel vectors (centres, lefts, rights, …) rather than
 *  a struct-of-arrays TreeNode.  The per-query traversal stack is allocated
 *  locally instead of being thread_local.
 * ════════════════════════════════════════════════════════════════════════════ */
static CandidateList epipolar_wedge_filter_with_segment_tree_origin(
    const Points &kp1, const Points &kp2,
    const Eigen::Ref<const Eigen::Matrix3f> &F,
    float radius)
{
    constexpr float MAX_ANGLE = 180.f;
    constexpr float kRad2Deg  = 180.f / float(M_PI);

    /* epipole in image 2 */
    const Eigen::Vector3f ev = computeEpipole(F.transpose());
    const float ex = ev(0), ey = ev(1);

    /* build angular interval lists for kp2 */
    std::vector<float> start_ang, end_ang;
    std::vector<int>   own;
    std::vector<int>   index_pts_in_disk;

    const int N2 = static_cast<int>(kp2.size());
    start_ang.reserve(2 * N2);
    end_ang  .reserve(2 * N2);
    own      .reserve(2 * N2);

    auto add = [&](float a0, float a1, int j) {
        if (a0 <= a1) {
            start_ang.push_back(a0); end_ang.push_back(a1);        own.push_back(j);
        } else {
            start_ang.push_back(a0); end_ang.push_back(MAX_ANGLE); own.push_back(j);
            start_ang.push_back(0.f); end_ang.push_back(a1);       own.push_back(j);
        }
    };

    for (int j = 0; j < N2; ++j) {
        float a1, a2;
        if (!compute_tangent_angles(ex, ey, kp2.x[j], kp2.y[j], radius, a1, a2, MAX_ANGLE)) {
            index_pts_in_disk.push_back(j);
            continue;
        }
        add(a1, a2, j);
    }

    /* build segment tree (separate-vector layout) */
    const std::size_t N = start_ang.size();
    std::vector<float> centres, perm_st(N), perm_en(N);
    std::vector<int>   lefts, rights, perm_own(N);
    std::vector<std::pair<int, int>>     lohi;
    std::vector<std::pair<float, float>> max_min_ranges;
    int wp = 0;

    {
        std::vector<int> all(N);
        std::iota(all.begin(), all.end(), 0);
        std::vector<std::tuple<std::vector<int>, int, bool>> S;
        S.emplace_back(std::move(all), -1, false);

        while (!S.empty()) {
            auto [sub, parent, isR] = S.back();
            S.pop_back();
            if (sub.empty()) {
                if (parent >= 0) {
                    if (isR) rights[parent] = -1;
                    else     lefts[parent]  = -1;
                }
                continue;
            }
            std::vector<float> mids(sub.size());
            for (std::size_t i = 0; i < sub.size(); ++i)
                mids[i] = (start_ang[sub[i]] + end_ang[sub[i]]) * 0.5f;
            std::nth_element(mids.begin(), mids.begin() + mids.size() / 2, mids.end());
            const float mid = mids[sub.size() / 2];

            std::vector<int> here, L, R;
            for (int k : sub) {
                if      (end_ang[k]   < mid) L.push_back(k);
                else if (start_ang[k] > mid) R.push_back(k);
                else                          here.push_back(k);
            }
            const int lo        = wp;
            float min_range     = std::numeric_limits<float>::max();
            float max_range     = std::numeric_limits<float>::lowest();
            for (int k : here) {
                perm_st[wp]    = start_ang[k];
                perm_en[wp]    = end_ang[k];
                perm_own[wp++] = own[k];
                max_range      = std::max(max_range, end_ang[k]);
                min_range      = std::min(min_range, start_ang[k]);
            }
            const int idx = static_cast<int>(centres.size());
            centres.push_back(mid);
            lohi.emplace_back(lo, wp);
            max_min_ranges.emplace_back(min_range, max_range);
            lefts.push_back(-1);
            rights.push_back(-1);
            if (parent >= 0) {
                if (isR) rights[parent] = idx;
                else     lefts[parent]  = idx;
            }
            S.emplace_back(std::move(R), idx, true);
            S.emplace_back(std::move(L), idx, false);
        }
    }

    /* query: appends matching owner indices into buf (does not clear it) */
    auto query = [&](float ang, std::vector<int> &buf) {
        if (centres.empty()) return;
        std::vector<int> stk = {0};
        while (!stk.empty()) {
            const int n = stk.back(); stk.pop_back();
            const auto [lo, hi]           = lohi[n];
            const auto [min_r, max_r]     = max_min_ranges[n];
            if (ang >= min_r && ang <= max_r)
                for (int i = lo; i < hi; ++i)
                    if (perm_st[i] <= ang && ang <= perm_en[i])
                        buf.push_back(perm_own[i]);
            if (ang < centres[n] && lefts[n]  != -1) stk.push_back(lefts[n]);
            if (ang > centres[n] && rights[n] != -1) stk.push_back(rights[n]);
        }
    };

    /* query each kp1[i] */
    const int N1 = static_cast<int>(kp1.size());
    CandidateList cand(N1);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N1; ++i) {
        std::vector<int> buf;
        buf.insert(buf.end(), index_pts_in_disk.begin(), index_pts_in_disk.end());

        const float p1x = kp1.x[i], p1y = kp1.y[i];
        const float lx  = F(0,0)*p1x + F(0,1)*p1y + F(0,2);
        const float ly  = F(1,0)*p1x + F(1,1)*p1y + F(1,2);
        const float ang = wrap_angle(std::atan2(-lx, ly) * kRad2Deg, MAX_ANGLE);

        query(ang, buf);
        cand[i] = std::move(buf);
    }

    return cand;
}


/* ════════════════════════════════════════════════════════════════════════════
 *  SEGMENT-TREE EPIPOLAR WEDGE FILTER
 *
 *  1.  Compute the right epipole e in image 2.
 *  2.  For each kp2[j] compute the angular interval [a1,a2] (tangent wedge
 *      from e to a tolerance circle of radius R around kp2[j]).
 *  3.  Build a binary segment tree over those intervals.
 *  4.  For each kp1[i] compute the epipolar direction angle ang and query the
 *      tree for all intervals that contain ang.  O(log N + K) per query.
 * ════════════════════════════════════════════════════════════════════════════ */

// All per-node metadata in one struct → fits in ~one cache line per node
// instead of requiring 5 independent cache-line loads.
struct TreeNode {
    float centre;
    float min_range, max_range;
    int lo, hi;       // range into the flat interval arrays ps/pe/po
    int left, right;  // child indices (-1 = none)
};

static CandidateList epipolar_wedge_filter_with_segment_tree(
    const Points &kp1, const Points &kp2,
    const Eigen::Ref<const Eigen::Matrix3f> &F,
    float radius)
{
    /* ── epipole in image 2: right null-vector of F^T ────────────────────── */
    // JacobiSVD is entirely stack-allocated for 3x3; no heap alloc, no copy.
    const Eigen::Vector3f ev = computeEpipole(F.transpose());
    const float ex = ev(0), ey = ev(1);

    /* ── build angular interval lists for kp2 ────────────────────────────── */
    constexpr float MAX_ANGLE = 180.f;
    constexpr float kRad2Deg  = 180.f / float(M_PI);
    const int N2 = static_cast<int>(kp2.size());

    std::vector<float> start_ang, end_ang;
    std::vector<int>   own;
    std::vector<int>   index_pts_in_disk;

    // Each point produces at most 2 intervals (wrap-around); reserve upfront.
    start_ang.reserve(2 * N2);
    end_ang  .reserve(2 * N2);
    own      .reserve(2 * N2);
    index_pts_in_disk.reserve(N2);

    auto add_interval = [&](float a0, float a1, int j) {
        if (a0 <= a1) {
            start_ang.push_back(a0); end_ang.push_back(a1); own.push_back(j);
        } else {
            start_ang.push_back(a0); end_ang.push_back(MAX_ANGLE); own.push_back(j);
            start_ang.push_back(0.f); end_ang.push_back(a1);       own.push_back(j);
        }
    };

    for (int j = 0; j < N2; ++j) {
        float a1, a2;
        if (!compute_tangent_angles(ex, ey, kp2.x[j], kp2.y[j], radius, a1, a2, MAX_ANGLE)) {
            index_pts_in_disk.push_back(j);
            continue;
        }
        add_interval(a1, a2, j);
    }

    /* ── build segment tree ──────────────────────────────────────────────── */
    const std::size_t N = start_ang.size();

    // Flat storage for interval data referenced by node [lo, hi) ranges.
    std::vector<float> ps(N), pe(N);
    std::vector<int>   po(N);
    int wp = 0;

    // All nodes in one contiguous array for cache-friendly traversal.
    std::vector<TreeNode> nodes;
    nodes.reserve(2 * N + 1);

    // Scratch buffers hoisted out of the recursion to eliminate O(N log N)
    // heap allocations (one set per node in the original code).
    std::vector<float> mids;
    std::vector<int>   here, L, R;
    mids.reserve(N);
    here.reserve(N); L.reserve(N); R.reserve(N);

    // Iterative build via explicit stack.
    std::vector<std::tuple<std::vector<int>, int, bool>> S;
    {
        std::vector<int> all(N);
        std::iota(all.begin(), all.end(), 0);
        S.emplace_back(std::move(all), -1, false);
    }

    while (!S.empty()) {
        auto [sub, parent, isR] = std::move(S.back());
        S.pop_back();

        if (sub.empty()) {
            if (parent >= 0) {
                if (isR) nodes[parent].right = -1;
                else     nodes[parent].left  = -1;
            }
            continue;
        }

        mids.resize(sub.size());
        for (std::size_t i = 0; i < sub.size(); ++i)
            mids[i] = (start_ang[sub[i]] + end_ang[sub[i]]) * 0.5f;
        std::nth_element(mids.begin(), mids.begin() + mids.size() / 2, mids.end());
        const float mid = mids[sub.size() / 2];

        here.clear(); L.clear(); R.clear();
        for (int k : sub) {
            if      (end_ang[k]   < mid) L.push_back(k);
            else if (start_ang[k] > mid) R.push_back(k);
            else                          here.push_back(k);
        }

        const int lo = wp;
        float min_range = std::numeric_limits<float>::max();
        float max_range = std::numeric_limits<float>::lowest();
        for (int k : here) {
            ps[wp] = start_ang[k];
            pe[wp] = end_ang[k];
            po[wp++] = own[k];
            min_range = std::min(min_range, start_ang[k]);
            max_range = std::max(max_range, end_ang[k]);
        }

        const int idx = static_cast<int>(nodes.size());
        nodes.push_back({mid, min_range, max_range, lo, wp, -1, -1});

        if (parent >= 0) {
            if (isR) nodes[parent].right = idx;
            else     nodes[parent].left  = idx;
        }

        S.emplace_back(std::move(R), idx, true);
        S.emplace_back(std::move(L), idx, false);
    }

    /* ── query: collect all intervals containing `ang` ───────────────────── */
    // Raw pointers prove non-aliasing to the compiler.
    const TreeNode *__restrict nd  = nodes.data();
    const float    *__restrict psd = ps.data();
    const float    *__restrict ped = pe.data();
    const int      *__restrict pod = po.data();

    // Appends matching owner indices directly into buf; stk is a reusable
    // scratch buffer passed in by the caller (thread_local in the hot loop).
    auto query = [&](float ang, std::vector<int> &buf, std::vector<int> &stk) {
        if (nodes.empty()) return;
        stk.clear();
        stk.push_back(0);
        while (!stk.empty()) {
            const int n = stk.back(); stk.pop_back();
            if (ang >= nd[n].min_range && ang <= nd[n].max_range)
                for (int i = nd[n].lo; i < nd[n].hi; ++i)
                    if (psd[i] <= ang && ang <= ped[i])
                        buf.push_back(pod[i]);
            if (ang < nd[n].centre && nd[n].left  != -1) stk.push_back(nd[n].left);
            if (ang > nd[n].centre && nd[n].right != -1) stk.push_back(nd[n].right);
        }
    };

    /* ── query each kp1[i] ───────────────────────────────────────────────── */
    const int N1 = static_cast<int>(kp1.size());
    CandidateList cand(N1);

    // Pre-scalar F values — avoids Eigen operator() in the hot loop.
    const float F00 = F(0,0), F01 = F(0,1), F02 = F(0,2);
    const float F10 = F(1,0), F11 = F(1,1), F12 = F(1,2);

    // schedule(static): query cost is log(N)+K — fairly uniform across
    // iterations (same tree, similar K).  Static avoids dynamic's per-chunk
    // atomic and has lower thread-barrier overhead.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N1; ++i)
    {
        // Both vectors are thread_local: allocated once per OS thread,
        // reused across all N1 iterations → zero heap alloc per query.
        thread_local std::vector<int> stk;
        thread_local std::vector<int> buf;

        buf.clear();
        // Reserve capacity heuristic: thread_local buf keeps its allocated
        // capacity from the previous iteration, so after warmup this is free.
        // The first few calls may reallocate; the reserve below caps that.
        if (buf.capacity() < index_pts_in_disk.size() + 32)
            buf.reserve(index_pts_in_disk.size() + 128);
        if (!index_pts_in_disk.empty())
            buf.insert(buf.end(), index_pts_in_disk.begin(), index_pts_in_disk.end());

        const float p1x = kp1.x[i], p1y = kp1.y[i];
        const float lx  = F00*p1x + F01*p1y + F02;
        const float ly  = F10*p1x + F11*p1y + F12;
        const float ang = wrap_angle(std::atan2(-lx, ly) * kRad2Deg, MAX_ANGLE);

        query(ang, buf, stk);

        cand[i] = buf;   // copy (buf keeps its capacity for the next iteration)
    }

    return cand;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  ANGULAR HASH EPIPOLAR FILTER  (optimised)
 *
 *  1.  Assign each kp2[j] to an angular bucket using scalar F^T arithmetic
 *      (no Eigen matrix-vector call in the hot loop).
 *  2.  Store buckets as a CSR flat array: contiguous x/y coordinates per
 *      bucket enable auto-vectorised distance checks with no gather loads.
 *  3.  Query phase filters directly into cand[i] — no per-query tmp vector,
 *      no intermediate heap allocation, no extra copy.
 * ════════════════════════════════════════════════════════════════════════════ */
static CandidateList epipolar_hash_filter_cpp(
    const Points &kp1, const Points &kp2,
    const Eigen::Ref<const Eigen::Matrix3f> &F,
    int   bins             = 45,
    float tol_px           = 3.f,
    bool  use_sampson      = false,
    bool  probe_adjacent   = true,
    bool  distance_filtering = true)
{
    const Eigen::Vector3f e1 = computeEpipole(F);
    const float e1x = e1(0), e1y = e1(1);

    const int N1 = static_cast<int>(kp1.size());
    const int N2 = static_cast<int>(kp2.size());

    constexpr float kSpan = float(M_PI);
    const float bins_over_span = static_cast<float>(bins) / kSpan;

    // Precompute F^T scalar coefficients for the bucket-angle formula:
    //   l = F^T * [x, y, 1]^T  →  theta = atan2(l[0], -l[1])
    //   l[0] = F(0,0)*x + F(1,0)*y + F(2,0)
    //   l[1] = F(0,1)*x + F(1,1)*y + F(2,1)
    const float Ft00 = F(0,0), Ft01 = F(1,0), Ft02 = F(2,0);
    const float Ft10 = F(0,1), Ft11 = F(1,1), Ft12 = F(2,1);

    // ── Step 1: assign each kp2[j] to a bucket ───────────────────────────────
    std::vector<int> bkt_of(N2);
    std::vector<int> cnt(bins, 0);
    for (int j = 0; j < N2; ++j) {
        float lx    = Ft00 * kp2.x[j] + Ft01 * kp2.y[j] + Ft02;
        float ly    = Ft10 * kp2.x[j] + Ft11 * kp2.y[j] + Ft12;
        float theta = std::atan2(lx, -ly);
        if (theta < 0.f) theta += kSpan;
        int k = static_cast<int>(theta * bins_over_span);
        if (k >= bins) k = bins - 1;
        bkt_of[j] = k;
        ++cnt[k];
    }

    // ── Step 2: CSR offsets ───────────────────────────────────────────────────
    std::vector<int> off(bins + 1, 0);
    for (int b = 0; b < bins; ++b) off[b + 1] = off[b] + cnt[b];

    // ── Step 3: scatter kp2 into flat contiguous arrays by bucket ────────────
    // flat_x / flat_y are contiguous per bucket → inner distance loop sees
    // sequential loads, no gather → compiler can auto-vectorise with SIMD.
    std::vector<float> flat_x(N2), flat_y(N2);
    std::vector<int>   flat_orig(N2);
    {
        std::vector<int> pos(off.begin(), off.begin() + bins);  // fill cursors
        for (int j = 0; j < N2; ++j) {
            int p        = pos[bkt_of[j]]++;
            flat_x[p]    = kp2.x[j];
            flat_y[p]    = kp2.y[j];
            flat_orig[p] = j;
        }
    }

    // Raw pointers prove non-aliasing to the compiler.
    const float *__restrict fx   = flat_x.data();
    const float *__restrict fy   = flat_y.data();
    const int   *__restrict fo   = flat_orig.data();
    const int   *__restrict boff = off.data();

    const int probe_start = probe_adjacent ? -1 : 0;
    const int probe_end   = probe_adjacent ?  1 : 0;

    // F row scalars for the epipolar line per kp1[i] query
    const float F00 = F(0,0), F01 = F(0,1), F02 = F(0,2);
    const float F10 = F(1,0), F11 = F(1,1), F12 = F(1,2);
    const float F20 = F(2,0), F21 = F(2,1), F22 = F(2,2);

    // ── Step 4: query each kp1[i] ────────────────────────────────────────────
    //
    // Key vectorisation insight:
    //   The brute-force's inner loop (N2 = 30 000 iterations) vectorises at
    //   SIMD width because it is one large contiguous scan.  Three separate
    //   loops of N2/bins each (~667) do NOT vectorise as well because the trip
    //   count is too small for the compiler to amortise the SIMD setup cost.
    //
    //   Fix (fast path, ~96 % of queries): when the three probed buckets do
    //   NOT wrap around the circular [0, bins) space, they sit side-by-side
    //   in the CSR flat array → one contiguous range of 3 × N2/bins elements.
    //   A two-phase loop then lets the compiler generate a pure SIMD phase 1
    //   (FMA + compare, no side-effects) and a compact scalar phase 2
    //   (push_back only for the ~K ≪ N2/bins passing elements).
    //
    //   Edge-case path (k0 = 0 or k0 = bins-1): three separate loops as
    //   before – affects only 2/bins ≈ 4 % of queries.

    CandidateList cand(N1);
    const float la_den_scale = tol_px * tol_px;  // reused below

    // Work per iteration is ~uniform (each query scans the same ~3*N2/bins
    // elements) → schedule(static) has lower synchronisation overhead than
    // schedule(dynamic).
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N1; ++i) {
        float tp = std::atan2(kp1.y[i] - e1y, kp1.x[i] - e1x);
        if (tp < 0.f) tp += kSpan;
        int k0 = static_cast<int>(tp * bins_over_span);
        if (k0 >= bins) k0 = bins - 1;

        auto &out = cand[i];

        if (!distance_filtering) {
            for (int d = probe_start; d <= probe_end; ++d) {
                int b = (k0 + d + bins) % bins;
                out.insert(out.end(), fo + boff[b], fo + boff[b + 1]);
            }
            continue;
        }

        const float la = F00*kp1.x[i] + F01*kp1.y[i] + F02;
        const float lb = F10*kp1.x[i] + F11*kp1.y[i] + F12;
        const float lc = F20*kp1.x[i] + F21*kp1.y[i] + F22;

        if (use_sampson) {
            // Sampson — rarely used, kept scalar.
            for (int d = probe_start; d <= probe_end; ++d) {
                int b  = (k0 + d + bins) % bins;
                int lo = boff[b], hi = boff[b + 1];
                for (int p = lo; p < hi; ++p) {
                    const float x2 = fx[p], y2 = fy[p];
                    const float num  = la*x2 + lb*y2 + lc;
                    const float g0   = Ft00*x2 + Ft01*y2 + Ft02;
                    const float g1   = Ft10*x2 + Ft11*y2 + Ft12;
                    const float den  = la*la + lb*lb + g0*g0 + g1*g1;
                    if (den > 0.f && num*num <= tol_px*tol_px * den)
                        out.push_back(fo[p]);
                }
            }
            continue;
        }

        const float tol_sq_norm = la_den_scale * (la*la + lb*lb);

        // ── FAST PATH: no wrap-around → one contiguous CSR range ─────────
        // Covers k0 ∈ [1, bins-2]  (≈ (bins-2)/bins of all queries).
        if (probe_adjacent && k0 > 0 && k0 < bins - 1) {
            const int lo = boff[k0 - 1];
            const int hi = boff[k0 + 2];   // boff has bins+1 entries
            const int P  = hi - lo;

            // Two-phase filter — pure SIMD phase 1, scalar compact phase 2.
            //   kTLBufSz = 4096: mask=4 KB + tl_buf=16 KB = 20 KB per thread
            //   → fits in 32 KB Zen4 L1-D.
            //   Overflow guard: if this bucket cluster is unusually large
            //   (P ≥ kTLBufSz), fall back to direct push_back.
            constexpr int kTLBufSz = 1 << 12;  // 4096
            thread_local static uint8_t mask[kTLBufSz];
            thread_local static int     tl_buf[kTLBufSz];

            if (P < kTLBufSz) {
                const float *__restrict fxp = fx + lo;
                const float *__restrict fyp = fy + lo;
                const int   *__restrict fop = fo + lo;
                int n_out = 0;

                // Phase 1: SIMD (no side-effects, no aliasing).
#pragma omp simd simdlen(16)
                for (int p = 0; p < P; ++p) {
                    const float r = la * fxp[p] + lb * fyp[p] + lc;
                    mask[p] = static_cast<uint8_t>(r * r <= tol_sq_norm);
                }
                // Phase 2: compact.
                for (int p = 0; p < P; ++p) {
                    if (mask[p]) tl_buf[n_out++] = fop[p];
                }
                out.insert(out.end(), tl_buf, tl_buf + n_out);
            } else {
                // Rare overflow: scan contiguous range with direct push_back.
                for (int p = lo; p < hi; ++p) {
                    const float r = la * fx[p] + lb * fy[p] + lc;
                    if (r * r <= tol_sq_norm) out.push_back(fo[p]);
                }
            }

        } else {
            // ── EDGE/SCALAR PATH: wrap-around or no adjacency probing ────
            for (int d = probe_start; d <= probe_end; ++d) {
                int b  = (k0 + d + bins) % bins;
                int lo = boff[b], hi = boff[b + 1];
                for (int p = lo; p < hi; ++p) {
                    const float r = la * fx[p] + lb * fy[p] + lc;
                    if (r * r <= tol_sq_norm) out.push_back(fo[p]);
                }
            }
        }
    }

    return cand;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  PYTHON BINDINGS
 * ════════════════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(guided_desc_match, m)
{
    m.doc() = "Pixel-accurate epipolar guided matching (segment-tree + brute-force)";

    // ── epipolar_geometric_distance_filter ───────────────────────────────────
    m.def(
        "epipolar_geometric_distance_filter",
        [](const py::array_t<float> &kp1_arr,
           const py::array_t<float> &kp2_arr,
           const Eigen::Matrix3f    &F,
           float tol)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto cand = epipolar_geometric_distance_filter(
                from_numpy(kp1_arr), from_numpy(kp2_arr), F, tol);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return std::make_tuple(cand, elapsed_ms);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("tolerance"),
        "Brute-force epipolar distance filter.\n"
        "Args:\n"
        "  kp1, kp2   : (N,2) float32 numpy arrays\n"
        "  F          : 3x3 fundamental matrix (numpy float32)\n"
        "  tolerance  : pixel tolerance\n"
        "Returns: (List[List[int]], float) – candidate kp2 indices for each kp1, elapsed ms");

    // ── epipolar_wedge_filter_with_segment_tree_origin ───────────────────────
    m.def(
        "epipolar_wedge_filter_with_segment_tree_origin",
        [](const py::array_t<float> &kp1_arr,
           const py::array_t<float> &kp2_arr,
           const Eigen::Matrix3f    &F,
           float radius)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto cand = epipolar_wedge_filter_with_segment_tree_origin(
                from_numpy(kp1_arr), from_numpy(kp2_arr), F, radius);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return std::make_tuple(cand, elapsed_ms);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("radius"),
        "Epipolar wedge filter using a segment tree (original separate-vectors layout).\n"
        "Args:\n"
        "  kp1, kp2   : (N,2) float32 numpy arrays\n"
        "  F          : 3x3 fundamental matrix (numpy float32)\n"
        "  radius     : tangent-wedge radius (pixel tolerance)\n"
        "Returns: (List[List[int]], float) – candidate kp2 indices for each kp1, elapsed ms");

    // ── epipolar_hash_filter_cpp ─────────────────────────────────────────────
    m.def(
        "epipolar_hash_filter_cpp",
        [](const py::array_t<float> &kp1_arr,
           const py::array_t<float> &kp2_arr,
           const Eigen::Matrix3f    &F,
           int   bins,
           float tol,
           bool  use_sampson,
           bool  probe_adjacent,
           bool  distance_filtering)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto cand = epipolar_hash_filter_cpp(
                from_numpy(kp1_arr), from_numpy(kp2_arr), F,
                bins, tol, use_sampson, probe_adjacent, distance_filtering);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return std::make_tuple(cand, elapsed_ms);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"),
        py::arg("bins")              = 45,
        py::arg("tol")               = 3.f,
        py::arg("use_sampson")       = false,
        py::arg("probe_adjacent")    = true,
        py::arg("distance_filtering")= true,
        "Angular-hash epipolar filter.  O(N/bins + K) per query.\n"
        "Args:\n"
        "  kp1, kp2          : (N,2) float32 numpy arrays\n"
        "  F                 : 3x3 fundamental matrix (numpy float32)\n"
        "  bins              : number of angular hash buckets (default 45)\n"
        "  tol               : pixel tolerance for distance refinement\n"
        "  use_sampson       : use Sampson distance instead of point-line\n"
        "  probe_adjacent    : also probe neighbouring buckets\n"
        "  distance_filtering: refine candidates with epipolar distance\n"
        "Returns: (List[List[int]], float) – candidate indices, elapsed ms");

    // ── epipolar_wedge_filter_with_segment_tree ──────────────────────────────
    m.def(
        "epipolar_wedge_filter_with_segment_tree",
        [](const py::array_t<float> &kp1_arr,
           const py::array_t<float> &kp2_arr,
           const Eigen::Matrix3f    &F,
           float radius)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto cand = epipolar_wedge_filter_with_segment_tree(
                from_numpy(kp1_arr), from_numpy(kp2_arr), F, radius);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return std::make_tuple(cand, elapsed_ms);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("radius"),
        "Epipolar wedge filter using a segment tree.  O(log N + K) per query.\n"
        "Args:\n"
        "  kp1, kp2   : (N,2) float32 numpy arrays\n"
        "  F          : 3x3 fundamental matrix (numpy float32)\n"
        "  radius     : tangent-wedge radius (pixel tolerance)\n"
        "Returns: (List[List[int]], float) – candidate kp2 indices for each kp1, elapsed ms");
}
