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

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N1; ++i)
    {
        // thread_local: the stack vector is allocated once per OS thread and
        // reused across all queries on that thread — zero heap alloc per call.
        thread_local std::vector<int> stk;

        std::vector<int> buf;
        buf.reserve(index_pts_in_disk.size() + 16);
        buf.insert(buf.end(), index_pts_in_disk.begin(), index_pts_in_disk.end());

        const float p1x = kp1.x[i], p1y = kp1.y[i];
        const float lx  = F(0,0)*p1x + F(0,1)*p1y + F(0,2);
        const float ly  = F(1,0)*p1x + F(1,1)*p1y + F(1,2);
        const float ang = wrap_angle(std::atan2(-lx, ly) * kRad2Deg, MAX_ANGLE);

        // buf is passed directly — no intermediate tree_hits copy.
        query(ang, buf, stk);

        cand[i] = std::move(buf);
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
