// guided_desc_match_dense.cpp
#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>

// #include <omp.h>
// omp_set_num_threads(1);

namespace py = pybind11;
#define PRINT_VAR(matrix) std::cout << #matrix << " \n" << matrix << std::endl;

#define PRINT_MATRIX_SIZE(matrix)                                                  \
    std::cout << #matrix << ".cols() size: " << (matrix).cols() << ", " << #matrix \
              << ".rows() size: " << (matrix).rows() << std::endl;
/* ───────────────────────────── type aliases ────────────────────────────── */
using PointList     = std::vector<std::tuple<float, float>>;
using CandidateList = std::vector<std::vector<int>>;  // dense, fast(nah)
using Clock         = std::chrono::high_resolution_clock;

enum class FilterType {
    None,       // no filtering
    GMSTest,    // GMS outlier removal
    RatioTest,  // Lowe's ratio test
};

class FilterConfig {
   public:
    FilterType type = FilterType::None;

    // Lowe's ratio test parameters

    float ratio_threshold   = 0.75f;  // Lowe's default
    bool use_adaptive_ratio = false;  // adaptive ratio test
    bool apply_ratio_test   = true;   // whether to apply the ratio test

    // GMS parameters
    int min_votes = 6;  // minimum votes per cell-pair in GMS

    // other parameters
    bool force_all_matches_return = false;  // force all matches, no filtering

    FilterConfig() = default;
};

class MatchingConfig {
   public:
    bool execute_LUT = false;  // whether to execute the LUT-based epipolar filter
    float radius     = 10.f;   // radius for the LUT-based epipolar filter

    bool execute_epipolar_distance_filter =
        false;                                // whether to execute the geometric distance filter
    float epipolar_distance_tolerance = 3.f;  // tolerance for the geometric distance filter

    bool execute_hash_filter      = true;  // whether to execute the epipolar hashing filter
    int bins                      = 45;    // number of bins for the epipolar hashing filter
    float epipolar_hash_tolerance = 3.f;   // tolerance for the epipolar hashing filter
    bool use_sampson =
        false;  // whether to use the Sampson distance for the epipolar hashing filter
    bool probe_adjacent = true;  // whether to probe adjacent cells in the epipolar hashing filter
    bool distance_filtering =
        true;  // whether to apply distance filtering in the epipolar hashing filter
    std::optional<std::pair<int, int>> shape1 = std::nullopt;  // shape of the first image
    std::optional<std::pair<int, int>> shape2 = std::nullopt;  // shape of the second image

    bool execute_grid_search_filter = true;  // whether to execute the staggered grid search filter
    float grid_cell_size = 32.f;  // size of the grid cells for the staggered grid search filter
    float grid_step_size = 20.f;  // step size for sampling along the epipolar lines
    float grid_tolerance = 3.f;   // tolerance for the staggered grid search filter
    bool use_sampson_grid =
        false;  // whether to use the Sampson distance for the staggered grid search filter
    bool probe_adjacent_grid =
        true;  // whether to probe adjacent cells in the staggered grid search filter
    bool distance_filtering_grid =
        true;  // whether to apply distance filtering in the staggered grid search filter
    std::optional<std::pair<int, int>> shape_grid1 =
        std::nullopt;  // shape of the first image for grid search
    std::optional<std::pair<int, int>> shape_grid2 =
        std::nullopt;  // shape of the second image for grid search

    bool execute_segment_tree_filter = true;  // whether to execute the wedge + segment tree filter
    float segment_tree_radius        = 10.f;  // radius for the wedge + segment tree filter
};

/* ========= utility ======================================================= */
inline float wrap_angle_deg(float a, float max_angle = 360.f) {
    a = std::fmod(a, max_angle);
    return (a < 0) ? a + max_angle : a;
}
inline float mod_positive(float x, float p) {
    float r = std::fmod(x, p);
    return (r < 0) ? r + p : r;
}

inline float line_angle(const Eigen::Vector3f& l) {  // [0,π)
    float th = std::atan2(l(0), -l(1));
    return (th < 0.f) ? th + float(M_PI) : th;
}

inline float point_line_dist(const std::tuple<float, float>& p, const Eigen::Vector3f& l) {
    float a = l(0), b = l(1), c = l(2);
    return std::abs(a * std::get<0>(p) + b * std::get<1>(p) + c) / std::sqrt(a * a + b * b);
}

inline void push_unique(CandidateList& C, int i, int j) {
    auto& v = C[i];
    if (v.empty() || v.back() != j) v.push_back(j);
}

/* ----- dict ↔ dense list helpers (keep Python API unchanged) ------------- */
inline CandidateList dict2list(py::dict d, std::size_t N) {
    CandidateList L(N);
    for (auto& kv : d) {
        int i = kv.first.cast<int>();
        L[i]  = kv.second.cast<std::vector<int>>();
    }
    return L;
}
inline py::dict list2dict(const CandidateList& L) {
    py::dict d;
    for (std::size_t i = 0; i < L.size(); ++i)
        if (!L[i].empty()) d[py::int_(i)] = py::cast(L[i]);
    return d;
}

template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
as_eigen_colmajor(const py::array_t<T>& arr) {
    if (arr.flags() & py::array::f_style) {
        // Already Fortran order, can map directly
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
            arr.data(), arr.shape(0), arr.shape(1));
    } else {
        // Not Fortran order, need to copy to column-major
        py::array_t<T, py::array::f_style> arr_f = py::array_t<T, py::array::f_style>(arr);
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
            arr_f.data(), arr_f.shape(0), arr_f.shape(1));
    }
}

// --- conversion helpers (needed by the bindings below) ---
static PointList convert_tuple_list(const std::vector<std::tuple<float, float>>& in) {
    PointList out;
    out.reserve(in.size());
    for (auto& t : in) out.emplace_back(std::get<0>(t), std::get<1>(t));
    return out;
}

static PointList convert_tuple_list(const py::array_t<float>& in) {
    PointList out;
    auto eigen_map = as_eigen_colmajor(in);

    out.reserve(eigen_map.rows());
    for (int i = 0; i < eigen_map.rows(); ++i) {
        out.emplace_back(eigen_map(i, 0), eigen_map(i, 1));
    }
    return out;
}

static py::dict convert_list_to_dict(const std::vector<std::vector<int>>& cand) {
    py::dict d;
    for (size_t i = 0; i < cand.size(); ++i) {
        if (!cand[i].empty()) d[py::int_(i)] = py::cast(cand[i]);
    }
    return d;
}

Eigen::Vector3f epipole_via_lu(const Eigen::Matrix3f& F) {
    Eigen::FullPivLU<Eigen::Matrix3f> lu(F);
    // kernel() returns a basis for the null‐space as columns of a matrix
    Eigen::Matrix<float, 3, 1> kern = lu.kernel();
    // kernel() might have dimension >1 if F is more degenerate, but normally it's 1
    Eigen::Vector3f e = kern.col(0);
    return e / e.z();
}

/* ======================================================================== */
/*  Tangent wedge routine (unchanged)                                       */
/* ======================================================================== */
inline bool compute_tangent_angles(const std::tuple<float, float>& P,
                                   const std::tuple<float, float>& C, float R, float& a1, float& a2,
                                   float max_angle = 360.f) {
    float dx = std::get<0>(P) - std::get<0>(C);
    float dy = std::get<1>(P) - std::get<1>(C);
    float d2 = dx * dx + dy * dy;
    if (d2 <= R * R) return false;
    float l  = R * R / d2;
    float m  = R * std::sqrt(d2 - R * R) / d2;
    float vx = l * dx, vy = l * dy;
    float px = -m * dy, py = m * dx;
    float t1x = std::get<0>(C) + vx + px, t1y = std::get<1>(C) + vy + py;
    float t2x = std::get<0>(C) + vx - px, t2y = std::get<1>(C) + vy - py;
    a1 = wrap_angle_deg(
        std::atan2(t1y - std::get<1>(P), t1x - std::get<0>(P)) * 180.f / float(M_PI), max_angle);
    a2 = wrap_angle_deg(
        std::atan2(t2y - std::get<1>(P), t2x - std::get<0>(P)) * 180.f / float(M_PI), max_angle);
    if (wrap_angle_deg(a2 - a1, max_angle) > 180.f) std::swap(a1, a2);
    return true;
}

/// Compute the right epipole (F·e = 0) from a 3×3 Eigen F using OpenCV’s solveZ.
Eigen::Vector3f computeRightEpipole(const Eigen::Matrix3f& F) {
    // 1) Copy Eigen→cv::Mat
    cv::Mat F_cv(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) F_cv.at<float>(i, j) = F(i, j);

    // 2) Solve null-space: F_cv * e_cv = 0
    cv::Mat e_cv;
    cv::SVD::solveZ(F_cv, e_cv);  // 3×1 CV_32F

    // 3) Normalize homogeneous coordinate
    e_cv /= e_cv.at<float>(2, 0);

    // 4) Convert back to Eigen
    Eigen::Vector3f e;
    cv::cv2eigen(e_cv, e);
    return e;
}

// -----------------------------------------------------------------------------
// Brute-force epipolar distance filter
//   – keeps every kp₂ whose geometric distance to the epipolar line of kp₁
//     is below the pixel tolerance
//   – SAME logic as the original version, but returns CandidateList
// -----------------------------------------------------------------------------
CandidateList epipolar_geometric_distance_filter(const PointList& kp1, const PointList& kp2,
                                                 const Eigen::Ref<const Eigen::Matrix3f>& F,
                                                 float tolerance_px) {
    constexpr float kEPS = 1e-8f;

    /* -------------- allocate output (N₁ rows) --------------------- */
    CandidateList cand(kp1.size());
    for (auto& v : cand) v.reserve(8);  // small pre-reserve

/* -------------- main double loop ------------------------------ */
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        /* epipolar line l2 of kp1i in image-2 ---------------------- */
        Eigen::Vector3f p1h(std::get<0>(kp1[i]), std::get<1>(kp1[i]), 1.f);
        Eigen::Vector3f l = F * p1h;
        float a = l(0), b = l(1), c = l(2);
        float denom = std::sqrt(a * a + b * b) + kEPS;

        /* test every kp2j ----------------------------------------- */
        for (std::size_t j = 0; j < kp2.size(); ++j) {
            float x = std::get<0>(kp2[j]);
            float y = std::get<1>(kp2[j]);
            float d = std::abs(a * x + b * y + c) / denom;

            if (d <= tolerance_px) cand[i].push_back(static_cast<int>(j));
        }
    }
    return cand;
}

/* ======================================================================== */
/*  LUT-based epipolar filter (dense)                                       */
/* ======================================================================== */
CandidateList lut_epipolar_filter(const PointList& kp1, const PointList& kp2,
                                  const Eigen::Ref<const Eigen::Matrix3f>& F, float radius) {
    bool consider_near_epi             = true;
    Eigen::Vector3f e                  = computeRightEpipole(F.transpose());
    const std::tuple<float, float> epi = {e(0), e(1)};

    /* epipolar angles for kp₁ ------------------------------------------------ */
    std::vector<float> ang(kp1.size());
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        Eigen::Vector3f p(std::get<0>(kp1[i]), std::get<1>(kp1[i]), 1.f);
        Eigen::Vector3f l = F * p;
        ang[i]            = wrap_angle_deg(std::atan2(-l(0), l(1)) * 180.f / float(M_PI));
    }
    std::vector<int> order(kp1.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return ang[a] < ang[b]; });
    std::vector<float> ang_sorted;
    ang_sorted.reserve(kp1.size());
    for (int idx : order) ang_sorted.push_back(ang[idx]);

    /* output container ------------------------------------------------------- */
    CandidateList cand(kp1.size());
    for (auto& v : cand) v.reserve(8);  // cheap heuristic

    /* iterate kp2 ------------------------------------------------------------ */
    std::vector<std::mutex> cand_mutex(kp1.size());
#pragma omp parallel for schedule(dynamic)
    for (std::size_t j = 0; j < kp2.size(); ++j) {
        auto range = [&](float lo, float hi) -> std::vector<int> {
            std::vector<int> tmp;
            tmp.reserve(order.size());
            if (lo <= hi) {
                auto it1 = std::lower_bound(ang_sorted.begin(), ang_sorted.end(), lo);
                auto it2 = std::upper_bound(ang_sorted.begin(), ang_sorted.end(), hi);
                for (auto it = it1; it != it2; ++it) tmp.push_back(order[it - ang_sorted.begin()]);
            } else {
                auto it1 = std::lower_bound(ang_sorted.begin(), ang_sorted.end(), lo);
                for (auto it = it1; it != ang_sorted.end(); ++it)
                    tmp.push_back(order[it - ang_sorted.begin()]);
                auto it2 = std::upper_bound(ang_sorted.begin(), ang_sorted.end(), hi);
                for (auto it = ang_sorted.begin(); it != it2; ++it)
                    tmp.push_back(order[it - ang_sorted.begin()]);
            }
            tmp.shrink_to_fit();
            return tmp;
        };

        float a1, a2;
        if (!compute_tangent_angles(epi, kp2[j], radius, a1, a2)) {
            if (consider_near_epi) {
                for (std::size_t i = 0; i < kp1.size(); ++i) {
                    std::lock_guard<std::mutex> lock(cand_mutex[i]);
                    cand[i].push_back(j);
                }
                continue;
            }
        }

        for (int i : range(a1, a2)) {
            std::lock_guard<std::mutex> lock(cand_mutex[i]);
            cand[i].push_back(j);
        }
        for (int i : range(wrap_angle_deg(a1 + 180), wrap_angle_deg(a2 + 180))) {
            std::lock_guard<std::mutex> lock(cand_mutex[i]);
            cand[i].push_back(j);
        }
    }
    return cand;
}

/* ======================================================================== */
/*  Guided descriptor ratio-test (dense inside)                             */
/* ======================================================================== */
inline float adaptive_ratio(int k) {
    if (k < 2) return 1.f;
    const float A = (0.9f - 0.45f) / (std::log(8000.f) - std::log(5.f));
    const float B = 0.9f - A * std::log(8000.f);
    return std::clamp(A * std::log(float(k)) + B, 0.4f, 0.95f);
}

// ----------------------------------------------------------------------------
// squared‐Lowe ratio test with optional adaptive correction
inline bool passes_ratio_test(float bestDist, float secondDist, int count_snn, float ratio_thresh,
                              bool apply_ratio_test, bool use_adaptive_ratio) {
    if (!apply_ratio_test) return true;
    if (count_snn <= 1) return true;  // single candidate always passes

    // base Lowe threshold²
    const float base_th_sq = ratio_thresh * ratio_thresh;

    if (use_adaptive_ratio) {
        // correction factor based on how many neighbors we saw
        float corr = 1.0f;
        if (count_snn < 3)
            corr = 0.25f * 0.25f;
        else if (count_snn < 5)
            corr = 0.5f * 0.5f;
        else if (count_snn < 10)
            corr = 0.6f * 0.6f;
        else if (count_snn < 20)
            corr = 0.65f * 0.65f;

        // corrected squared‐ratio
        float ratio_sq = (bestDist / secondDist) / corr;
        return ratio_sq < base_th_sq;
    } else {
        // standard Lowe test (squared)
        return (bestDist / secondDist) < base_th_sq;
    }
}

// ----------------------------------------------------------------------------
// BF_matching: brute‐force matching between two sets of descriptors

std::vector<cv::DMatch> BF_matching(
    py::array_t<float, py::array_t<float>::c_style | py::array_t<float>::forcecast> descriptors1,
    py::array_t<float, py::array_t<float>::c_style | py::array_t<float>::forcecast> descriptors2,
    float ratio_thresh = 0.75f, bool apply_ratio_test = true, bool use_adaptive_ratio = false) {
    py::buffer_info buf = descriptors1.request();

    if (buf.ndim != 2) throw std::runtime_error("Expected a 2D array");

    auto n_rows = buf.shape[0];
    auto n_cols = buf.shape[1];

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        desc_mat1(static_cast<float*>(buf.ptr), n_rows, n_cols);
    py::buffer_info buf2 = descriptors2.request();

    auto n_rows2 = buf2.shape[0];
    auto n_cols2 = buf2.shape[1];
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        desc_mat2(static_cast<float*>(buf2.ptr), n_rows2, n_cols2);

    std::vector<cv::DMatch> matches;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_rows; ++i) {
        const Eigen::VectorXf& desc1 = desc_mat1.row(i);
        float bestDist               = std::numeric_limits<float>::infinity();
        float secondDist             = std::numeric_limits<float>::infinity();
        int bestIdx                  = -1;

        for (size_t j = 0; j < n_rows2; ++j) {
            const Eigen::VectorXf& desc2 = desc_mat2.row(j);
            float dist2                  = (desc1 - desc2).squaredNorm();  // squared distance

            if (dist2 < bestDist) {
                secondDist = bestDist;
                bestDist   = dist2;
                bestIdx    = j;
            } else if (dist2 < secondDist) {
                secondDist = dist2;
            }
        }

        if (bestIdx >= 0 && passes_ratio_test(bestDist, secondDist, 1, ratio_thresh,
                                              apply_ratio_test, use_adaptive_ratio)) {
#pragma omp critical
            matches.emplace_back(int(i), bestIdx, 0, std::sqrt(bestDist));
        }
    }

    return matches;
}

// ----------------------------------------------------------------------------
// guided_match_ratio_test with optional one‐to‐one enforcement
std::vector<cv::DMatch> guided_match_ratio_test(
    py::array_t<float, py::array_t<float>::c_style | py::array_t<float>::forcecast> descriptors1,
    py::array_t<float, py::array_t<float>::c_style | py::array_t<float>::forcecast> descriptors2,
    const CandidateList& candidate_list, float ratio_thresh = 0.8f, bool apply_ratio_test = true,
    bool use_adaptive_ratio = false) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // grab raw pointers
    auto b1         = descriptors1.request();
    auto b2         = descriptors2.request();
    const float* d1 = static_cast<float*>(b1.ptr);
    const float* d2 = static_cast<float*>(b2.ptr);
    int D           = b1.shape[1];

    std::vector<cv::DMatch> matches;
    matches.reserve(candidate_list.size());

    // // 1) find best & second‐best for each queryIdx
    // #pragma omp parallel for schedule(dynamic)
    //     for (size_t i = 0; i < candidate_list.size(); ++i) {
    //         const auto& cands = candidate_list[i];
    //         if (cands.empty()) continue;

    //         const float* desc1 = d1 + i * D;
    //         float bestDist     = std::numeric_limits<float>::infinity();
    //         float secondDist   = std::numeric_limits<float>::infinity();
    //         int bestIdx        = -1;
    //         int count_snn      = 0;
    //         Eigen::Map<const Eigen::VectorXf> vec1(desc1, D);

    //         for (int j : cands) {
    //             const float* desc2 = d2 + j * D;
    //             Eigen::Map<const Eigen::VectorXf> vec2(desc2, D);
    //             float dist2 = (vec1 - vec2).squaredNorm();
    //             ++count_snn;
    //             if (dist2 >= secondDist)
    //                 continue;
    //             if (dist2 < bestDist) {
    //                 secondDist = bestDist;
    //                 bestDist   = dist2;
    //                 bestIdx    = j;
    //             } else if (dist2 < secondDist) {
    //                 secondDist = dist2;
    //             }
    //         }

    //         if (bestIdx >= 0 && passes_ratio_test(bestDist, secondDist, count_snn, ratio_thresh,
    //                                               apply_ratio_test, use_adaptive_ratio)) {
    // // critical section for thread‐safety
    // #pragma omp critical
    //             matches.emplace_back(int(i), bestIdx, 0, std::sqrt(bestDist));
    //         }
    //     }

    std::vector<std::vector<cv::DMatch>> local_matches(omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < candidate_list.size(); ++i) {
        const auto& cands = candidate_list[i];
        if (cands.empty()) continue;

        int tid = omp_get_thread_num();  // thread-local

        const float* desc1 = d1 + i * D;
        float bestDist     = std::numeric_limits<float>::infinity();
        float secondDist   = std::numeric_limits<float>::infinity();
        Eigen::Map<const Eigen::VectorXf> vec1(desc1, D);
        int bestIdx   = -1;
        int count_snn = 0;

        for (int j : cands) {
            const float* desc2 = d2 + j * D;

            Eigen::Map<const Eigen::VectorXf> vec2(desc2, D);
            float dist2 = (vec1 - vec2).squaredNorm();

            ++count_snn;
            if (dist2 < bestDist) {
                secondDist = bestDist;
                bestDist   = dist2;
                bestIdx    = j;
            } else if (dist2 < secondDist) {
                secondDist = dist2;
            }
        }

        if (bestIdx >= 0 && passes_ratio_test(bestDist, secondDist, count_snn, ratio_thresh,
                                              apply_ratio_test, use_adaptive_ratio)) {
            local_matches[tid].emplace_back(int(i), bestIdx, 0, std::sqrt(bestDist));
        }
    }

    // merge results
    for (const auto& lm : local_matches) {
        matches.insert(matches.end(), lm.begin(), lm.end());
    }

    // 2) enforce a true one‐to‐one by doing a little ratio test on trainIdx
    const bool enforce_unique = false;  // toggle for development
    if (enforce_unique) {
        struct Best2 {
            cv::DMatch best;
            cv::DMatch second;
            bool has_second = false;
        };
        std::unordered_map<int, Best2> best_per_train;

        // gather best & runner‐up per trainIdx
        for (auto& m : matches) {
            auto& b = best_per_train[m.trainIdx];
            if (b.best.trainIdx < 0) {
                // first time
                b.best = m;
            } else {
                if (m.distance < b.best.distance) {
                    b.second     = b.best;
                    b.best       = m;
                    b.has_second = true;
                } else if (!b.has_second || m.distance < b.second.distance) {
                    b.second     = m;
                    b.has_second = true;
                }
            }
        }

        // now re‐filter by squared‐ratio on the train side
        std::vector<cv::DMatch> unique_matches;
        unique_matches.reserve(best_per_train.size());
        const float side_th_sq = ratio_thresh * ratio_thresh;

        for (auto& kv : best_per_train) {
            auto& b = kv.second;
            // keep singletons or those passing side‐ratio test
            if (!b.has_second ||
                (b.best.distance * b.best.distance) / (b.second.distance * b.second.distance) <
                    side_th_sq) {
                unique_matches.push_back(b.best);
            }
        }

        matches.swap(unique_matches);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[TIMER] Guided ratio test took "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
    //           << " ms, found " << matches.size() << " matches.\n";

    return matches;
}

// ----------------------------------------------------------------------------
// guided_GMS_test: best‐descriptor + GMS outlier removal (no ratio test)
//
//   kp1, kp2         : keypoints in image1 / image2
//   descriptors1,2   : (N×D) float32 arrays of SIFT descriptors
//   candidate_list   : for each i in [0..kp1.size()), a vector of possible j's
//   min_votes        : minimum GMS votes per cell‐pair (default = 6)
//
// returns: filtered list of cv::DMatch (one‐to‐one by construction)
inline std::vector<cv::DMatch> guided_GMS_test(
    const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
    py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
    const CandidateList& candidate_list, int min_votes = 6) {
    // 1) pick *one* best‐descriptor match per queryIdx
    auto b1 = descriptors1.request(), b2 = descriptors2.request();
    const float* d1 = static_cast<float*>(b1.ptr);
    const float* d2 = static_cast<float*>(b2.ptr);
    int D           = b1.shape[1];

    std::vector<cv::DMatch> matches;
    matches.reserve(candidate_list.size());

    // #pragma omp parallel for schedule(dynamic)
    //     for (size_t i = 0; i < candidate_list.size(); ++i) {
    //         const auto& cands = candidate_list[i];
    //         if (cands.empty()) continue;
    //         const float* desc1 = d1 + i * D;
    //         float bestDist     = std::numeric_limits<float>::infinity();
    //         int bestIdx        = -1;
    //         Eigen::Map<const Eigen::VectorXf> vec1(desc1, D);

    //         for (int j : cands) {
    //             const float* desc2 = d2 + j * D;
    //             Eigen::Map<const Eigen::VectorXf> vec2(desc2, D);
    //             float dist2        = (vec1 - vec2).squaredNorm();
    //             if (dist2 < bestDist) {
    //                 bestDist = dist2;
    //                 bestIdx  = j;
    //             }
    //         }
    //         if (bestIdx >= 0) {
    // #pragma omp critical
    //             matches.emplace_back(int(i), bestIdx, 0, std::sqrt(bestDist));
    //         }
    //     }
    std::vector<std::vector<cv::DMatch>> local_matches(omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < candidate_list.size(); ++i) {
        const auto& cands = candidate_list[i];
        if (cands.empty()) continue;
        int tid = omp_get_thread_num();

        const float* desc1 = d1 + i * D;
        float bestDist     = std::numeric_limits<float>::infinity();
        int bestIdx        = -1;
        Eigen::Map<const Eigen::VectorXf> vec1(desc1, D);

        for (int j : cands) {
            const float* desc2 = d2 + j * D;

            Eigen::Map<const Eigen::VectorXf> vec2(desc2, D);
            float dist2 = (vec1 - vec2).squaredNorm();
            if (dist2 < bestDist) {
                bestDist = dist2;
                bestIdx  = j;
            }
        }

        if (bestIdx >= 0) {
            local_matches[tid].emplace_back(int(i), bestIdx, 0, std::sqrt(bestDist));
        }
    }

    // Merge all
    for (auto& lm : local_matches) matches.insert(matches.end(), lm.begin(), lm.end());

    // 2) build GMS vote table on a 20×20 grid
    constexpr int Gx = 20, Gy = 20;
    // infer image bounds from keypoint extents
    float max_x1 = 0, max_y1 = 0, max_x2 = 0, max_y2 = 0;
    for (auto& k : kp1) {
        max_x1 = std::max(max_x1, k.pt.x);
        max_y1 = std::max(max_y1, k.pt.y);
    }
    for (auto& k : kp2) {
        max_x2 = std::max(max_x2, k.pt.x);
        max_y2 = std::max(max_y2, k.pt.y);
    }
    int W1 = int(std::ceil(max_x1)) + 1, H1 = int(std::ceil(max_y1)) + 1;
    int W2 = int(std::ceil(max_x2)) + 1, H2 = int(std::ceil(max_y2)) + 1;

    auto grid_xy = [&](float x, float y, int W, int H) {
        int gx = std::min(Gx - 1, std::max(0, int(x * Gx / W)));
        int gy = std::min(Gy - 1, std::max(0, int(y * Gy / H)));
        return std::make_pair(gx, gy);
    };

    // votes[g1x][g1y][g2x][g2y]
    std::vector<int> votes(Gx * Gy * Gx * Gy, 0);
    auto idx4 = [&](int a, int b, int c, int d) { return (((a * Gy + b) * Gx + c) * Gy + d); };

    for (auto& m : matches) {
        auto [g1x, g1y] = grid_xy(kp1[m.queryIdx].pt.x, kp1[m.queryIdx].pt.y, W1, H1);
        auto [g2x, g2y] = grid_xy(kp2[m.trainIdx].pt.x, kp2[m.trainIdx].pt.y, W2, H2);
        votes[idx4(g1x, g1y, g2x, g2y)]++;
    }

    // 3) filter: keep only matches whose cell‐pair has ≥ min_votes
    std::vector<cv::DMatch> filtered;
    filtered.reserve(matches.size());
    for (auto& m : matches) {
        auto [g1x, g1y] = grid_xy(kp1[m.queryIdx].pt.x, kp1[m.queryIdx].pt.y, W1, H1);
        auto [g2x, g2y] = grid_xy(kp2[m.trainIdx].pt.x, kp2[m.trainIdx].pt.y, W2, H2);
        if (votes[idx4(g1x, g1y, g2x, g2y)] >= min_votes) filtered.push_back(m);
    }

    // 3) enforce a hard one‐to‐one (keep only the best per trainIdx)
    const bool enforce_unique = false;
    if (enforce_unique) {
        // Top 1 Version
        /*
        std::unordered_map<int, cv::DMatch> best_per_train;
        for (auto &m : filtered) {
            auto it = best_per_train.find(m.trainIdx);
            if (it == best_per_train.end() || m.distance < it->second.distance)
                best_per_train[m.trainIdx] = m;
        }
        std::vector<cv::DMatch> unique_matches;
        unique_matches.reserve(best_per_train.size());
        for (auto &kv : best_per_train)
            unique_matches.push_back(kv.second);
        return unique_matches;*/
        // count occurrences of each trainIdx

        // 2 - Remove any ambiguous matches version
        /*std::unordered_map<int,int> cnt;
        for (auto &m : filtered)
            ++cnt[m.trainIdx];

        // keep only those that occur exactly once
        std::vector<cv::DMatch> unique_matches;
        unique_matches.reserve(filtered.size());
        for (auto &m : filtered) {
            if (cnt[m.trainIdx] == 1)
                unique_matches.push_back(m);
        }
        return unique_matches;*/

        // Version Lowe's test on uniqueness matches
        struct Best2 {
            cv::DMatch best, second;
            bool has_second = false;
        };
        std::unordered_map<int, Best2> per_train;

        // gather best & runner‐up per trainIdx
        for (auto& m : filtered) {
            auto& b = per_train[m.trainIdx];
            if (b.best.trainIdx < 0) {
                b.best = m;
            } else {
                if (m.distance < b.best.distance) {
                    b.second     = b.best;
                    b.best       = m;
                    b.has_second = true;
                } else if (!b.has_second || m.distance < b.second.distance) {
                    b.second     = m;
                    b.has_second = true;
                }
            }
        }

        // re‐filter by Lowe's squared‐ratio on the train side
        std::vector<cv::DMatch> unique_matches;
        unique_matches.reserve(per_train.size());
        const float side_th_sq = 0.95f * 0.95f;  // standard Lowe = 0.8

        for (auto& kv : per_train) {
            auto& b = kv.second;
            if (!b.has_second ||
                (b.best.distance * b.best.distance) / (b.second.distance * b.second.distance) <
                    side_th_sq) {
                unique_matches.push_back(b.best);
            }
        }
        return unique_matches;
    }

    // 4) final 95%-percentile cutoff on distance
    const bool apply_percentile_cut = false;  // dev toggle
    if (apply_percentile_cut && !filtered.empty()) {
        // gather distances
        std::vector<float> ds;
        ds.reserve(filtered.size());
        for (auto& m : filtered) ds.push_back(m.distance);
        // find 95th percentile
        size_t k = size_t(0.70f * ds.size());
        std::nth_element(ds.begin(), ds.begin() + k, ds.end());
        float thresh = ds[k];
        // keep only the *best* 95%
        std::vector<cv::DMatch> final;
        final.reserve(filtered.size());
        for (auto& m : filtered)
            if (m.distance <= thresh) final.push_back(m);
        return final;
    }

    return filtered;
}

/* ========================================================================== */
/*  EPIPOLAR HASHING FILTER                                                        */
/* ========================================================================== */
CandidateList epipolar_hash_filter_cpp(const PointList& kp1, const PointList& kp2,
                                       const Eigen::Ref<const Eigen::Matrix3f>& F, int bins = 45,
                                       float tol_px = 3.f, bool use_sampson = false,
                                       bool probe_adjacent = true, bool distance_filtering = true,
                                       std::optional<std::pair<int, int>> shape1 = std::nullopt,
                                       std::optional<std::pair<int, int>> shape2 = std::nullopt) {
    auto t0 = std::chrono::high_resolution_clock::now();
    /* -------- epipole in image-1 ---------------------------------------- */
    Eigen::Vector3f e1 = computeRightEpipole(F);

    /* -------- image bounds --------------------------------------------- */
    int H1 = 0, W1 = 0, H2 = 0, W2 = 0;
    if (shape1) {
        H1 = shape1->first;
        W1 = shape1->second;
    } else
        for (auto& p : kp1) {
            H1 = std::max(H1, int(std::get<1>(p)) + 1);
            W1 = std::max(W1, int(std::get<0>(p)) + 1);
        }
    if (shape2) {
        H2 = shape2->first;
        W2 = shape2->second;
    } else
        for (auto& p : kp2) {
            H2 = std::max(H2, int(std::get<1>(p)) + 1);
            W2 = std::max(W2, int(std::get<0>(p)) + 1);
        }

    /* -------- angular span [a,a+π] (bow-tie) --------------------------- */
    bool inside = 0 <= e1(0) && e1(0) <= W1 && 0 <= e1(1) && e1(1) <= H1;
    float a = 0.f, span = float(M_PI);
    if (!inside) {
        const Eigen::Vector3f c[4] = {
            {0, 0, 1}, {float(W2), 0, 1}, {0, float(H2), 1}, {float(W2), float(H2), 1}};
        float ang[4];
        for (int i = 0; i < 4; ++i) ang[i] = line_angle(F.transpose() * c[i]);
        std::sort(ang, ang + 4);
        float ext[8];
        for (int i = 0; i < 4; ++i) {
            ext[i]     = ang[i];
            ext[i + 4] = ang[i] + float(M_PI);
        }
        int gap    = 0;
        float gmax = -1.f;
        for (int i = 0; i < 7; ++i) {
            float g = ext[i + 1] - ext[i];
            if (g > gmax) {
                gmax = g;
                gap  = i;
            }
        }
        a = std::fmod(ext[gap + 1], float(M_PI));
        if (a < 0) a += float(M_PI);
        span = float(M_PI);
    }

    /* -------- build buckets for kp₂ ----------------------------------- */
    std::vector<std::vector<int>> bucket(bins);

    for (auto& b : bucket) {
        b.reserve(kp2.size());  // small pre-reserve
    }

    std::vector<std::mutex> bucket_mutex(bins);
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < int(kp2.size()); ++j) {
        Eigen::Vector3f h(std::get<0>(kp2[j]), std::get<1>(kp2[j]), 1.f);
        float theta = line_angle(F.transpose() * h);
        int k       = int(std::floor(mod_positive(theta - a, span) * bins / span));
        std::lock_guard<std::mutex> lock(bucket_mutex[k]);
        bucket[k].push_back(j);
    }

    for (auto& b : bucket) {
        b.shrink_to_fit();
    }

    const std::vector<int> neigh =
        probe_adjacent ? std::vector<int>{-1, 0, 1} : std::vector<int>{0};

    /* -------- query kp₁ ------------------------------------------------ */
    CandidateList cand(kp1.size());
    for (auto& v : cand) v.reserve(8);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        float dx = std::get<0>(kp1[i]) - e1(0), dy = std::get<1>(kp1[i]) - e1(1);
        float tp = std::atan2(dy, dx);
        if (tp < 0) tp += float(M_PI);
        int k0 = int(std::floor(mod_positive(tp - a, span) * bins / span));

        std::vector<int> tmp;
        tmp.reserve(kp1.size());  // small pre-reserve
        for (int d : neigh) {
            const auto& b = bucket[(k0 + d + bins) % bins];
            tmp.insert(tmp.end(), b.begin(), b.end());
        }
        if (tmp.empty()) {
            continue;
        }

        if (!distance_filtering) {
            cand[i] = std::move(tmp);
            continue;
        }

        Eigen::Vector3f p1h(std::get<0>(kp1[i]), std::get<1>(kp1[i]), 1.f);
        Eigen::Vector3f l2 = F * p1h;

        for (int j : tmp) {
            bool ok;
            if (!use_sampson) {
                ok = point_line_dist(kp2[j], l2) <= tol_px;
            } else {
                Eigen::Vector3f p2h(std::get<0>(kp2[j]), std::get<1>(kp2[j]), 1.f);
                Eigen::Vector3f Fx1 = l2, Ft_p2 = F.transpose() * p2h;
                float num = std::pow(p2h.dot(Fx1), 2);
                float den = Fx1.head<2>().squaredNorm() + Ft_p2.head<2>().squaredNorm();
                ok        = den > 0 && num / den <= tol_px;
            }
            if (ok) cand[i].push_back(j);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[TIMER] Epipolar hash filter took "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
    //           << " ms, found " << cand.size() << " candidates.\n";
    return cand;
}

/* ========================================================================== */
/*  GRID-SEARCH FILTER                                                  */
/* ========================================================================== */
namespace {  // helper structs identical to the old version
struct FlatGrid {
    int rows, cols;

    // ToDo: check if all cells are occupied. If it is sparse, use a map instead.
    std::vector<std::vector<int>> cell;
    FlatGrid(int r = 0, int c = 0) : rows(r), cols(c), cell(size_t(r * c)) {}
    bool is_inside(int r, int c) const { return r >= 0 && r < rows && c >= 0 && c < cols; }
    std::vector<int>& at(int r, int c) { return cell[size_t(r) * cols + c]; }
    const std::vector<int>& at(int r, int c) const { return cell[size_t(r) * cols + c]; }
};
inline bool clip_line(const Eigen::Vector3f& l, int w, int h, Eigen::Vector2f& q0,
                      Eigen::Vector2f& q1) {
    float a = l(0), b = l(1), c = l(2);
    std::vector<Eigen::Vector2f> pts;
    auto push = [&](float x, float y) {
        if (0 <= x && x < w && 0 <= y && y < h) pts.emplace_back(x, y);
    };
    if (std::abs(b) > 1e-8f) {
        push(0, -(a * 0 + c) / b);
        push(float(w - 1), -(a * float(w - 1) + c) / b);
    }
    if (std::abs(a) > 1e-8f) {
        push(-(b * 0 + c) / a, 0);
        push(-(b * float(h - 1) + c) / a, float(h - 1));
    }
    if (pts.size() < 2) return false;
    q0 = pts[0];
    q1 = pts[1];
    return true;
}
inline void sample_seg(const Eigen::Vector2f& p0, const Eigen::Vector2f& p1, float step,
                       std::vector<Eigen::Vector2f>& out) {
    float d = (p1 - p0).norm();
    if (d < 1e-3f) return;
    int n               = std::max(1, int(std::floor(d / step)));
    Eigen::Vector2f dir = (p1 - p0) / d;
    for (int k = 0; k <= n; ++k) out.emplace_back(p0 + dir * (k * step));
}
}  // namespace

CandidateList epipolar_grid_search_filter_cpp(const PointList& kp1, const PointList& kp2,
                                              const Eigen::Ref<const Eigen::Matrix3f>& F,
                                              float cell, float step, float tol_px,
                                              bool use_sampson, bool probe_adjacent,
                                              bool distance_filtering,
                                              std::optional<std::pair<int, int>> shape2) {
    /* image-2 size ----------------------------------------------------- */
    int H2 = 0, W2 = 0;
    if (shape2) {
        H2 = shape2->first;
        W2 = shape2->second;
    } else
        for (auto& p : kp2) {
            H2 = std::max(H2, int(std::get<1>(p)) + 1);
            W2 = std::max(W2, int(std::get<0>(p)) + 1);
        }

    /* four staggered grids -------------------------------------------- */
    const std::array<std::pair<float, float>, 4> off = {
        {{0, 0}, {cell / 2, 0}, {0, cell / 2}, {cell / 2, cell / 2}}};
    auto ncols = [&](float ox) { return int(std::ceil((W2 - ox) / cell)); };
    auto nrows = [&](float oy) { return int(std::ceil((H2 - oy) / cell)); };

    std::array<FlatGrid, 4> G = {FlatGrid(nrows(off[0].second), ncols(off[0].first)),
                                 FlatGrid(nrows(off[1].second), ncols(off[1].first)),
                                 FlatGrid(nrows(off[2].second), ncols(off[2].first)),
                                 FlatGrid(nrows(off[3].second), ncols(off[3].first))};

    // #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < int(kp2.size()); ++j) {
        float x = std::get<0>(kp2[j]), y = std::get<1>(kp2[j]);
        for (int g = 0; g < 4; ++g) {
            const auto [ox, oy] = off[g];
            int c               = int(std::floor((x - ox) / cell));
            int r               = int(std::floor((y - oy) / cell));
            // for now i exluded parallelization here, the issue is that, if you want to do it, we
            // need to craete mutex for each grid cell, which is not very efficient
            if (G[g].is_inside(r, c)) G[g].at(r, c).push_back(j);
        }
    }
    const std::vector<std::pair<int, int>> nbh =
        probe_adjacent ? std::vector<std::pair<int, int>>{{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}}
                       : std::vector<std::pair<int, int>>{{0, 0}};

    /* output ----------------------------------------------------------- */
    CandidateList cand(kp1.size());
    for (auto& v : cand) v.reserve(8);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        // Compute epipolar line in image-2
        Eigen::Vector3f p1h(std::get<0>(kp1[i]), std::get<1>(kp1[i]), 1.f);
        Eigen::Vector3f l = F * p1h;

        // Find intersection with image-2 bounds
        Eigen::Vector2f q0, q1;
        if (!clip_line(l, W2, H2, q0, q1)) continue;
        std::vector<Eigen::Vector2f> samples;
        samples.reserve(64);
        sample_seg(q0, q1, step, samples);
        if (samples.empty()) continue;

        // Create a set of candidate indices
        std::unordered_set<int> s;
        s.reserve(32);
        for (const auto& pt : samples) {
            float bestd = FLT_MAX;
            int bestg = 0, bestr = 0, bestc = 0;
            // For each sample point, find the best grid cell
            for (int g = 0; g < 4; ++g) {
                auto [ox, oy] = off[g];
                int c         = int(std::floor((pt.x() - ox) / cell));
                int r         = int(std::floor((pt.y() - oy) / cell));
                if (!G[g].is_inside(r, c)) continue;
                float cx = ox + (c + 0.5f) * cell, cy = oy + (r + 0.5f) * cell;
                float d2 = (pt.x() - cx) * (pt.x() - cx) + (pt.y() - cy) * (pt.y() - cy);
                if (d2 < bestd) {
                    bestd = d2;
                    bestg = g;
                    bestr = r;
                    bestc = c;
                }
            }
            // If we found a valid grid cell, add its candidates
            for (auto [dr, dc] : nbh) {
                int r = bestr + dr, c = bestc + dc;
                if (G[bestg].is_inside(r, c)) {
                    const auto& vec = G[bestg].at(r, c);
                    s.insert(vec.begin(), vec.end());
                }
            }
        }
        if (s.empty()) continue;
        // If we are not filtering by distance, just assign the candidates
        if (!distance_filtering) {
            cand[i].assign(s.begin(), s.end());
            continue;
        }

        // Otherwise, filter candidates by distance to the epipolar line (Not needed for this
        // algorithm)
        Eigen::Vector3f Fx1 = l;
        for (int j : s) {
            bool ok;
            if (!use_sampson) {
                ok = point_line_dist(kp2[j], l) <= tol_px;
            } else {
                Eigen::Vector3f p2h(std::get<0>(kp2[j]), std::get<1>(kp2[j]), 1.f);
                Eigen::Vector3f Ft_p2 = F.transpose() * p2h;
                float num             = std::pow(p2h.dot(Fx1), 2);
                float den             = Fx1.head<2>().squaredNorm() + Ft_p2.head<2>().squaredNorm();
                ok                    = den > 0 && num / den <= tol_px;
            }
            if (ok) cand[i].push_back(j);
        }
    }
    return cand;
}

/* ========================================================================== */
/*   WEDGE/TREE FILTER (segment-tree)                                   */
/* ========================================================================== */
CandidateList epipolar_wedge_filter_with_segment_tree(const PointList& kp1, const PointList& kp2,
                                                      const Eigen::Ref<const Eigen::Matrix3f>& F,
                                                      float radius) {
    bool consider_near_epi = true;

    auto t_start = Clock::now();

    // PRINT_VAR(F);

    /* epipole ---------------------------------------------------------- */
    Eigen::Vector3f e                  = computeRightEpipole(F.transpose());
    const std::tuple<float, float> epi = {e(0), e(1)};

    // PRINT_VAR(e);
    /* build interval lists -------------------------------------------- */
    std::vector<float> start_ang, end_ang;
    std::vector<int> own;

    const float MAX_ANGLE = 180.f;  // degrees
    auto add              = [&](float a0, float a1, int j) {
        if (a0 <= a1) {
            start_ang.push_back(a0);
            end_ang.push_back(a1);
            own.push_back(j);
        } else {
            start_ang.push_back(a0);
            end_ang.push_back(MAX_ANGLE);
            own.push_back(j);
            start_ang.push_back(0);
            end_ang.push_back(a1);
            own.push_back(j);
        }
    };

    auto t0 = Clock::now();
    std::vector<int> index_pts_in_disk;  // We will store the points in the circles
    for (int j = 0; j < int(kp2.size()); ++j) {
        float a1, a2;
        if (!compute_tangent_angles(epi, kp2[j], radius, a1, a2, MAX_ANGLE)) {
            index_pts_in_disk.push_back(j);
            continue;
        };
        add(a1, a2, j);
        // add(wrap_angle_deg(a1 + 180), wrap_angle_deg(a2 + 180), j);
    }
    auto t1 = Clock::now();

    std::cout << "[TIMER] Build angular intervals: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms\n";

    /* build simple array “segment tree” -------------------------------- */
    auto build_array_tree = [&](const auto& starts, const auto& ends, const auto& owners) {
        /* exactly as original, unchanged – returns tuple of vectors      */
        std::vector<float> centres, perm_st, perm_en;
        std::vector<int> lefts, rights, perm_own;
        std::vector<std::pair<int, int>> lohi;
        std::vector<std::pair<float, float>> max_min_ranges;
        std::vector<int> stck;
        std::vector<int> all(starts.size());
        std::iota(all.begin(), all.end(), 0);
        std::vector<std::tuple<std::vector<int>, int, bool>> S;
        // add reserves for vectors
        S.emplace_back(all, -1, false);
        int wp = 0;
        perm_st.resize(starts.size());
        perm_en.resize(starts.size());
        perm_own.resize(starts.size());

        while (!S.empty()) {
            auto [sub, parent, isR] = S.back();
            S.pop_back();
            if (sub.empty()) {
                if (parent >= 0) {
                    if (isR)
                        rights[parent] = -1;
                    else
                        lefts[parent] = -1;
                }
                continue;
            }
            std::vector<float> mids(sub.size());
            for (std::size_t i = 0; i < sub.size(); ++i)
                mids[i] = (starts[sub[i]] + ends[sub[i]]) * 0.5f;
            std::nth_element(mids.begin(), mids.begin() + mids.size() / 2, mids.end());
            float mid = mids[sub.size() / 2];
            std::vector<int> here, L, R;
            for (int k : sub) {
                if (ends[k] < mid)
                    L.push_back(k);
                else if (starts[k] > mid)
                    R.push_back(k);
                else
                    here.push_back(k);
            }
            int lo          = wp;
            float min_range = std::numeric_limits<float>::max();
            float max_range = std::numeric_limits<float>::lowest();
            for (int k : here) {
                perm_st[wp]    = starts[k];
                perm_en[wp]    = ends[k];
                perm_own[wp++] = owners[k];
                max_range      = std::max(max_range, ends[k]);
                min_range      = std::min(min_range, starts[k]);
            }
            int hi  = wp;
            int idx = centres.size();
            centres.push_back(mid);
            lohi.emplace_back(lo, hi);
            max_min_ranges.emplace_back(min_range, max_range);
            lefts.push_back(-1);
            rights.push_back(-1);
            if (parent >= 0) {
                if (isR)
                    rights[parent] = idx;
                else
                    lefts[parent] = idx;
            }
            S.emplace_back(R, idx, true);
            S.emplace_back(L, idx, false);
        }
        return std::make_tuple(centres, lefts, rights, lohi, max_min_ranges, perm_st, perm_en,
                               perm_own);
    };

    auto t2 = Clock::now();
    auto [centres, lefts, rights, lohi, max_min_ranges, ps, pe, po] =
        build_array_tree(start_ang, end_ang, own);

    auto t3 = Clock::now();
    std::cout << "[TIMER] Build segment tree: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms\n";

    auto query = [&](float ang, std::vector<int>& buf) {
        buf.clear();
        std::vector<int> S = {0};
        while (!S.empty()) {
            int n = S.back();
            S.pop_back();
            auto [lo, hi]               = lohi[n];
            auto [min_range, max_range] = max_min_ranges[n];
            if (ang >= min_range && ang <= max_range)
                for (int i = lo; i < hi; ++i)
                    if (ps[i] <= ang && ang <= pe[i]) buf.push_back(po[i]);
            if (ang < centres[n] && lefts[n] != -1) S.push_back(lefts[n]);
            if (ang > centres[n] && rights[n] != -1) S.push_back(rights[n]);
        }
    };

    /* query kp₁ -------------------------------------------------------- */
    CandidateList cand(kp1.size());
    // for (auto& v : cand) v.reserve(index_pts_in_disk.size() *2);

    auto t4 = Clock::now();

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        std::vector<int> buf;
        buf.reserve(kp1.size());

        if (consider_near_epi) {
            buf.insert(buf.end(), index_pts_in_disk.begin(), index_pts_in_disk.end());
        }
        Eigen::Vector3f p(std::get<0>(kp1[i]), std::get<1>(kp1[i]), 1.f);
        Eigen::Vector3f l = F * p;
        float ang = wrap_angle_deg(std::atan2(-l(0), l(1)) * 180.f / float(M_PI), MAX_ANGLE);
        int count = 0;
        query(ang, buf);
        cand[i] = std::move(buf);
        cand[i].shrink_to_fit();
        // if (consider_near_epi == true) {
        //     // add all the indexes of the points in the disk
        //     for (std::size_t k = 0; k < index_pts_in_disk.size(); ++k) {
        //         cand[i].push_back(index_pts_in_disk[k]);
        //     }
        // }
    }
    auto t5 = Clock::now();
    std::cout << "[TIMER] Query kp1: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << "ms\n";
    auto t6 = Clock::now();
    std::cout << "[TIMER] Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t_start).count()
              << " ms\n";
    std::cout << " Size of index_pts_in_disk: " << index_pts_in_disk.size() << "\n";

    return cand;
}

/* ========================================================================== */
/*   LUT-BINNED FILTER                                                  */
/* ========================================================================== */
CandidateList epipolar_lut_binned_filter_cpp(
    const PointList& kp1, const PointList& kp2, const Eigen::Ref<const Eigen::Matrix3f>& F,
    float tol_px, int bins = 120, const std::string& edge_mode = "geom", bool use_sampson = false,
    std::optional<std::pair<int, int>> shape2 = std::nullopt, bool debug = false) {
    bool consider_near_epi = true;
    /* ---- epipole in image-2 ----------------------------------------- */
    Eigen::Vector3f e2 = computeRightEpipole(F.transpose());
    const std::tuple<float, float> ref_t{e2(0), e2(1)};

    /* ---- wedge limits for kp2 -------------------------------------- */
    std::vector<float> a_min(kp2.size(), std::numeric_limits<float>::quiet_NaN()),
        a_max(kp2.size(), std::numeric_limits<float>::quiet_NaN());
    for (std::size_t j = 0; j < kp2.size(); ++j)
        compute_tangent_angles(ref_t, kp2[j], tol_px, a_min[j], a_max[j]);

    /* ---- epipolar angles for kp₁ ----------------------------------- */
    std::vector<float> ang1(kp1.size());
    std::vector<Eigen::Vector3f> p1h(kp1.size());
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        p1h[i]  = {std::get<0>(kp1[i]), std::get<1>(kp1[i]), 1.f};
        ang1[i] = line_angle(F * p1h[i]);
    }

    /* ---- angular span [a,a+π) -------------------------------------- */
    int H2 = 0, W2 = 0;
    if (shape2) {
        H2 = shape2->first;
        W2 = shape2->second;
    } else
        for (auto& p : kp2) {
            H2 = std::max(H2, int(std::get<1>(p)) + 1);
            W2 = std::max(W2, int(std::get<0>(p)) + 1);
        }
    Eigen::Vector3f C[4] = {
        {0, 0, 1}, {float(W2), 0, 1}, {0, float(H2), 1}, {float(W2), float(H2), 1}};
    float ac[4];
    for (int i = 0; i < 4; ++i) ac[i] = line_angle(F.transpose() * C[i]);
    std::sort(ac, ac + 4);
    float ext[8];
    for (int i = 0; i < 4; ++i) {
        ext[i]     = ac[i];
        ext[i + 4] = ac[i] + float(M_PI);
    }
    int gap  = 0;
    float gm = -1.f;
    for (int i = 0; i < 7; ++i) {
        float g = ext[i + 1] - ext[i];
        if (g > gm) {
            gm  = g;
            gap = i;
        }
    }
    float a = std::fmod(ext[gap + 1], float(M_PI)), span = float(M_PI);

    /* ---- buckets for kp1 ------------------------------------------- */
    std::vector<std::vector<int>> B(bins);
    std::vector<std::mutex> B_mutex(bins);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < kp1.size(); ++i) {
        int k = int(((ang1[i] - a) < 0 ? ang1[i] - a + span : ang1[i] - a) * bins / span);
        std::lock_guard<std::mutex> lock(B_mutex[k]);
        B[k].push_back(int(i));
    }

    /* ---- output ----------------------------------------------------- */
    CandidateList cand(kp1.size());
    for (auto& v : cand) v.reserve(8);
    bool do_geom = edge_mode != "none";

    auto test_dist = [&](int i, int j) -> bool {
        if (!do_geom) return true;
        if (!use_sampson) return point_line_dist(kp2[j], F * p1h[i]) <= tol_px;
        Eigen::Vector3f p2h(std::get<0>(kp2[j]), std::get<1>(kp2[j]), 1.f);
        Eigen::Vector3f Fx1 = F * p1h[i], Ft_p2 = F.transpose() * p2h;
        float num = std::pow(p2h.dot(Fx1), 2);
        float den = Fx1.head<2>().squaredNorm() + Ft_p2.head<2>().squaredNorm();
        return den > 0 && num / den <= tol_px;
    };

    std::vector<std::mutex> cand_mutex(kp1.size());
#pragma omp parallel for schedule(dynamic)
    for (std::size_t j = 0; j < kp2.size(); ++j) {
        if (std::isnan(a_min[j])) {
            if (consider_near_epi == true) {
                for (std::size_t i = 0; i < kp1.size(); ++i) {
                    cand[i].push_back(int(j));
                }  // push everything if the kp is inside the radius}
                continue;
            }
        }
        float lo = std::fmod(a_min[j] * float(M_PI) / 180.f, float(M_PI));
        float hi = std::fmod(a_max[j] * float(M_PI) / 180.f, float(M_PI));
        int k_lo = int(((lo - a) < 0 ? lo - a + span : lo - a) * bins / span);
        int k_hi = int(((hi - a) < 0 ? hi - a + span : hi - a) * bins / span);

        /* fully covered buckets -- add items*/
        if (k_lo != k_hi) {
            for (int k = (k_lo + 1) % bins; k != k_hi; k = (k + 1) % bins)
                for (int i : B[k]) {
                    std::lock_guard<std::mutex> lock(cand_mutex[i]);
                    cand[i].push_back(int(j));
                }
        }
        /* edges -- check geometric valid*/
        for (int k_edge : {k_lo, k_hi})
            for (int i : B[k_edge])
                if (test_dist(i, int(j))) {
                    std::lock_guard<std::mutex> lock(cand_mutex[i]);
                    push_unique(cand, i, int(j));
                }
    }

    if (debug) {
        std::size_t tot = 0;
        for (auto& v : cand) tot += v.size();
        std::cerr << "[LUT-binned] " << tot << " candidate pairs\n";
    }
    return cand;
}

std::tuple<float, float, float> compare_two_cands(const CandidateList& cands_gt,
                                                  const CandidateList& cands) {
    // std::cout << "Comparing two candidate lists...\n";
    int points_num = cands_gt.size();
    // std::cout << "Number of points: " << points_num << "\n";

    std::vector<float> precisions(points_num, 0.0f);
    std::vector<float> recalls(points_num, 0.0f);
    std::vector<float> outl(points_num, 0.0f);

    auto get_median = [](std::vector<float>& v) -> float {
        size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        return v[mid];
    };

    auto get_mean = [](const std::vector<float>& v) -> float {
        float sum = 0.0f;
        for (float val : v) {
            sum += val;
        }
        return sum / static_cast<float>(v.size());
    };

    for (std::size_t i = 0; i < cands_gt.size(); ++i) {
        const auto& gt   = cands_gt[i];
        const auto& test = cands[i];

        if (gt.empty() && test.empty()) {
            precisions[i] = 1.0f;
            recalls[i]    = 1.0f;
            outl[i]       = 0.0f;
            continue;
        }

        std::unordered_set<int> gt_set(gt.begin(), gt.end());
        std::unordered_set<int> test_set(test.begin(), test.end());

        int tp = 0;  // True Positives
        for (int idx : test_set) {
            if (gt_set.find(idx) != gt_set.end()) {
                ++tp;
            }
        }
        int fp = test_set.size() - tp;  // False Positives

        precisions[i] = (test_set.empty())
                            ? 1.0f
                            : static_cast<float>(tp) / static_cast<float>(test_set.size());
        recalls[i] =
            (gt_set.empty()) ? 0.0f : static_cast<float>(tp) / static_cast<float>(gt_set.size());
        outl[i] = (test_set.empty()) ? 0.0f
                                     : static_cast<float>(fp) / static_cast<float>(test_set.size());
    }
    float precision = get_mean(precisions);
    float recall    = get_mean(recalls);
    float outl_mean = get_mean(outl);
    // std::cout << "done comparing candidates.\n";
    return {precision, recall, outl_mean};
}

py::dict compare_cands_list(const py::dict& cands_gt, const py::dict& cands_test) {
    py::dict result;
    const CandidateList& cands1 = cands_gt["candidates"].cast<CandidateList>();
    const CandidateList& cands2 = cands_test["candidates"].cast<CandidateList>();

    std::cout << "Number of candidates in GT: " << cands1.size() << "\n";
    std::cout << "Number of candidates in Test: " << cands2.size() << "\n";

    return result;
}

py::dict evaluate_one_candidate_list(
    const PointList& kp1, const PointList& kp2,
    py::array_t<float, py::array::c_style | py::array::forcecast>& descriptors1,
    py::array_t<float, py::array::c_style | py::array::forcecast>& descriptors2,
    const CandidateList& cands, const FilterConfig& filter_config) {
    py::dict result;
    // Convert descriptors to Eigen matrices
    // You can add more evaluation metrics here

    auto t1 = std::chrono::high_resolution_clock::now();

    result["candidates"] = py::none();
    result["matches"]    = py::none();

    if (filter_config.type == FilterType::None) {
        result["candidates"] = py::cast(cands);  // No filtering, just return candidates
    } else if (filter_config.type == FilterType::GMSTest) {
        std::vector<cv::KeyPoint> kps1, kps2;
        kps1.reserve(kp1.size());
        for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
        kps2.reserve(kp2.size());
        for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
        result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2, cands,
                                                     filter_config.min_votes));
    } else if (filter_config.type == FilterType::RatioTest) {
        result["matches"] = py::cast(guided_match_ratio_test(
            descriptors1, descriptors2, cands, filter_config.ratio_threshold,
            filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
        // If we want to return all matches, we just return the candidates
        result["candidates"] = py::cast(cands);
    }

    auto t_filter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    result["filter_duration"] = py::int_(t_filter_duration);

    return result;
}

Eigen::VectorXi find_3d_matches_vec(const Eigen::MatrixXf& points_3d_1,
                                    const Eigen::MatrixXf& points_3d_2, const CandidateList& cands,
                                    float thr_3d) {
    PRINT_MATRIX_SIZE(points_3d_1);
    PRINT_MATRIX_SIZE(points_3d_2);
    Eigen::VectorXi local_matches = Eigen::VectorXi::Constant(points_3d_1.rows(), -1);
    float thr_3d_sq               = thr_3d * thr_3d;

#pragma omp parallel
    for (int i = 0; i < points_3d_1.rows(); ++i) {
        const auto& cand = cands[i];
        if (cand.empty()) continue;

        for (int j : cand) {
            if (j < 0 || j >= points_3d_2.rows()) continue;  // Check bounds
            float dist_sq = (points_3d_1.row(i) - points_3d_2.row(j)).squaredNorm();
            if (dist_sq <= thr_3d_sq) {
                local_matches(i) = j;
                break;
            }
        }
    }

    const Eigen::Index matched = (local_matches.array() >= 0).count();
    std::cout << "3D matches found: " << matched << " out of " << local_matches.rows() << "\n";

    return local_matches;
}

py::dict all_matches_combined(
    const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
    py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
    py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
    const py::array_t<float>& points_3d_1, const py::array_t<float>& points_3d_2,
    py::array_t<float>& F, const FilterConfig& filter_config, const MatchingConfig& matching_config,
    float thr_3d = 0.1f) {
    PointList kp1           = convert_tuple_list(kp1_tup);
    PointList kp2           = convert_tuple_list(kp2_tup);
    Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);
    py::dict total_result;

    const auto points_3d_eigen_1 = as_eigen_colmajor(points_3d_1);
    const auto points_3d_eigen_2 = as_eigen_colmajor(points_3d_2);

    CandidateList gt_candidates;

    auto get_ms_duration = [](auto t_start, auto t_end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    };

    if (matching_config.execute_epipolar_distance_filter) {
        // std::cout << "Executing epipolar distance filter...\n";
        auto t1       = std::chrono::high_resolution_clock::now();
        gt_candidates = epipolar_geometric_distance_filter(
            kp1, kp2, F_eigen, matching_config.epipolar_distance_tolerance);
        auto t2                 = std::chrono::high_resolution_clock::now();
        auto t_wedge_duration   = get_ms_duration(t1, t2);
        auto result             = evaluate_one_candidate_list(kp1, kp2, descriptors1, descriptors2,
                                                              gt_candidates, filter_config);
        result["cand_duration"] = py::int_(t_wedge_duration);

        auto [precision, recall, outl_mean] = compare_two_cands(gt_candidates, gt_candidates);
        result["precision"]                 = py::float_(precision);
        result["recall"]                    = py::float_(recall);
        result["outlier_mean"]              = py::float_(outl_mean);
        total_result["BF+Geo_combined"]     = std::move(result);

        total_result["3D_matches"] = py::cast(
            find_3d_matches_vec(points_3d_eigen_1, points_3d_eigen_2, gt_candidates, thr_3d));
    }

    if (matching_config.execute_LUT) {
        // std::cout << "Executing LUT binned filter...\n";
        auto t1         = std::chrono::high_resolution_clock::now();
        auto candidates = lut_epipolar_filter(kp1, kp2, F_eigen, matching_config.radius);
        // return py::dict("candidates"_a = std::move(candidates));
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t_cand_duration = get_ms_duration(t1, t2);
        auto result = evaluate_one_candidate_list(kp1, kp2, descriptors1, descriptors2, candidates,
                                                  filter_config);
        result["cand_duration"]             = py::int_(t_cand_duration);
        auto [precision, recall, outl_mean] = compare_two_cands(gt_candidates, candidates);
        result["precision"]                 = py::float_(precision);
        result["recall"]                    = py::float_(recall);
        result["outlier_mean"]              = py::float_(outl_mean);

        total_result["LUT_combined"] = std::move(result);
    }

    if (matching_config.execute_hash_filter) {
        // std::cout << "Executing epipolar hash filter...\n";
        auto t1         = std::chrono::high_resolution_clock::now();
        auto candidates = epipolar_hash_filter_cpp(
            kp1, kp2, F_eigen, matching_config.bins, matching_config.epipolar_hash_tolerance,
            matching_config.use_sampson, matching_config.probe_adjacent,
            matching_config.distance_filtering, matching_config.shape1, matching_config.shape2);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t_cand_duration = get_ms_duration(t1, t2);
        auto result = evaluate_one_candidate_list(kp1, kp2, descriptors1, descriptors2, candidates,
                                                  filter_config);
        result["cand_duration"] = py::int_(t_cand_duration);

        auto [precision, recall, outl_mean] = compare_two_cands(gt_candidates, candidates);
        result["precision"]                 = py::float_(precision);
        result["recall"]                    = py::float_(recall);
        result["outlier_mean"]              = py::float_(outl_mean);
        total_result["CPP_Hash_combined"]   = std::move(result);
    }

    if (matching_config.execute_grid_search_filter) {
        // std::cout << "Executing epipolar grid search filter...\n";
        auto t1         = std::chrono::high_resolution_clock::now();
        auto candidates = epipolar_grid_search_filter_cpp(
            kp1, kp2, F_eigen, matching_config.grid_cell_size, matching_config.grid_step_size,
            matching_config.grid_tolerance, matching_config.use_sampson_grid,
            matching_config.probe_adjacent_grid, matching_config.distance_filtering_grid,
            matching_config.shape_grid2);

        // epipolar_grid_search_filter_cpp(kp1, kp2, F_eigen, cell, step, tol, use_sampson,
        // probe_adjacent, distance_filtering, shape2);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t_cand_duration = get_ms_duration(t1, t2);
        auto result = evaluate_one_candidate_list(kp1, kp2, descriptors1, descriptors2, candidates,
                                                  filter_config);
        result["cand_duration"]             = py::int_(t_cand_duration);
        auto [precision, recall, outl_mean] = compare_two_cands(gt_candidates, candidates);
        result["precision"]                 = py::float_(precision);
        result["recall"]                    = py::float_(recall);
        result["outlier_mean"]              = py::float_(outl_mean);

        total_result["CPP_Grid_combined"] = std::move(result);
    }

    if (matching_config.execute_segment_tree_filter) {
        // std::cout << "Executing epipolar segment tree filter...\n";
        auto t1                      = std::chrono::high_resolution_clock::now();
        auto segment_tree_candidates = epipolar_wedge_filter_with_segment_tree(
            kp1, kp2, F_eigen, matching_config.segment_tree_radius);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t_bf_duration      = get_ms_duration(t1, t2);
        auto result             = evaluate_one_candidate_list(kp1, kp2, descriptors1, descriptors2,
                                                              segment_tree_candidates, filter_config);
        result["cand_duration"] = py::int_(t_bf_duration);

        auto [precision, recall, outl_mean] =
            compare_two_cands(gt_candidates, segment_tree_candidates);
        result["precision"]    = py::float_(precision);
        result["recall"]       = py::float_(recall);
        result["outlier_mean"] = py::float_(outl_mean);

        total_result["SegmentTree_combined"] = std::move(result);
    }

    // if()

    return total_result;
}

// m.def("all_matches_combined", &all_matches_combined, py::arg("kp1"), py::arg("kp2"),
// py::arg("descriptors1"), py::arg("descriptors2"),
//      py::arg("F"), py::arg("filter_config"), py::arg("matching_config"),
//       "Perform all matching strategies between two sets of descriptors.\n"
//       "This includes BF matching, epipolar wedge filtering, and any other defined
//       methods.\n");

// }  // namespace guided_desc_match

PYBIND11_MODULE(guided_desc_match, m) {
    m.doc() = "Epipolar‐guided descriptor matching";

    //     class FilterConfig {
    //    public:
    //     FilterType type = FilterType::None;

    //     // Lowe's ratio test parameters

    //     float ratio_threshold   = 0.75f;  // Lowe's default
    //     bool use_adaptive_ratio = false;  // adaptive ratio test
    //     bool apply_ratio_test   = true;   // whether to apply the ratio test

    //     // GMS parameters
    //     int min_votes = 6;  // minimum votes per cell-pair in GMS

    //     FilterConfig() = default;
    // };

    py::enum_<FilterType>(m, "FilterType")
        .value("None", FilterType::None)
        .value("GMSTest", FilterType::GMSTest)
        .value("RatioTest", FilterType::RatioTest)
        .export_values();

    py::class_<FilterConfig>(m, "FilterConfig")
        .def(py::init<>())  // default constructor
        .def(py::init<FilterType, float, bool, bool, int, bool>(), py::arg("type"),
             py::arg("ratio_threshold"), py::arg("use_adaptive_ratio") = false,
             py::arg("apply_ratio_test") = true, py::arg("min_votes") = 6,
             py::arg("force_all_matches_return") = false, "Configuration for filtering candidates.")
        .def_readwrite("type", &FilterConfig::type)
        .def_readwrite("threshold", &FilterConfig::ratio_threshold)
        .def_readwrite("use_adaptive_ratio", &FilterConfig::use_adaptive_ratio)
        .def_readwrite("apply_ratio_test", &FilterConfig::apply_ratio_test)
        .def_readwrite("min_votes", &FilterConfig::min_votes, "Minimum votes per cell-pair in GMS")
        .def_readwrite("force_all_matches_return", &FilterConfig::force_all_matches_return,
                       "If true, all matches are returned, even if they do not pass the filter");

    // --- cv::DMatch binding -------------------------------------------------
    py::class_<cv::DMatch>(m, "DMatch")
        .def(py::init<int, int, int, float>())
        .def_readwrite("queryIdx", &cv::DMatch::queryIdx)
        .def_readwrite("trainIdx", &cv::DMatch::trainIdx)
        .def_readwrite("imgIdx", &cv::DMatch::imgIdx)
        .def_readwrite("distance", &cv::DMatch::distance);

    py::class_<MatchingConfig>(m, "MatchingConfig")
        .def(py::init<>())

        // LUT-based epipolar filter
        .def_readwrite("execute_LUT", &MatchingConfig::execute_LUT,
                       "Whether to execute the LUT-based epipolar filter.")
        .def_readwrite("radius", &MatchingConfig::radius,
                       "Radius for the LUT-based epipolar filter.")

        // Epipolar distance filter
        .def_readwrite("execute_epipolar_distance_filter",
                       &MatchingConfig::execute_epipolar_distance_filter,
                       "Whether to execute the geometric distance filter.")
        .def_readwrite("epipolar_distance_tolerance", &MatchingConfig::epipolar_distance_tolerance,
                       "Tolerance for the geometric distance filter.")

        // Hash-based epipolar filter
        .def_readwrite("execute_hash_filter", &MatchingConfig::execute_hash_filter,
                       "Whether to execute the epipolar hashing filter.")
        .def_readwrite("bins", &MatchingConfig::bins,
                       "Number of bins for the epipolar hashing filter.")
        .def_readwrite("epipolar_hash_tolerance", &MatchingConfig::epipolar_hash_tolerance,
                       "Tolerance for the epipolar hashing filter.")
        .def_readwrite("use_sampson", &MatchingConfig::use_sampson,
                       "Whether to use the Sampson distance for the epipolar hashing filter.")
        .def_readwrite("probe_adjacent", &MatchingConfig::probe_adjacent,
                       "Whether to probe adjacent cells in the epipolar hashing filter.")
        .def_readwrite("distance_filtering", &MatchingConfig::distance_filtering,
                       "Whether to apply distance filtering in the epipolar hashing filter.")
        .def_readwrite("shape1", &MatchingConfig::shape1, "Shape of the first image.")
        .def_readwrite("shape2", &MatchingConfig::shape2, "Shape of the second image.")

        // Grid search filter
        .def_readwrite("execute_grid_search_filter", &MatchingConfig::execute_grid_search_filter,
                       "Whether to execute the staggered grid search filter.")
        .def_readwrite("grid_cell_size", &MatchingConfig::grid_cell_size, "Size of the grid cells.")
        .def_readwrite("grid_step_size", &MatchingConfig::grid_step_size,
                       "Step size for sampling along the epipolar lines.")
        .def_readwrite("grid_tolerance", &MatchingConfig::grid_tolerance,
                       "Tolerance for the grid search filter.")
        .def_readwrite("use_sampson_grid", &MatchingConfig::use_sampson_grid,
                       "Use Sampson distance in grid search filter.")
        .def_readwrite("probe_adjacent_grid", &MatchingConfig::probe_adjacent_grid,
                       "Probe adjacent cells in grid search.")
        .def_readwrite("distance_filtering_grid", &MatchingConfig::distance_filtering_grid,
                       "Use distance filtering in grid search.")
        .def_readwrite("shape_grid1", &MatchingConfig::shape_grid1,
                       "Shape of the first image for grid.")
        .def_readwrite("shape_grid2", &MatchingConfig::shape_grid2,
                       "Shape of the second image for grid.")

        // Segment tree filter
        .def_readwrite("execute_segment_tree_filter", &MatchingConfig::execute_segment_tree_filter,
                       "Whether to execute the segment tree filter.")
        .def_readwrite("segment_tree_radius", &MatchingConfig::segment_tree_radius,
                       "Radius for the segment tree filter.");

    m.def("convert_tuple_list", &compare_cands_list, py::arg("cands_gt"), py::arg("cands_test"),
          "Compare two candidate lists and return a dictionary with the results.\n");

    // --- guided_match_ratio_test --------------------------------------------
    m.def("guided_match_ratio_test", &guided_match_ratio_test, py::arg("descriptors1"),
          py::arg("descriptors2"), py::arg("candidate_list"), py::arg("ratio_thresh") = 0.75f,
          py::arg("use_adaptive_ratio") = false, py::arg("apply_ratio_test") = true,
          "Perform a ratio‐test guided by a precomputed candidate list.\n"
          "  descriptors1, descriptors2: (N×D) float arrays\n"
          "  candidate_list: List[List[int]] of size N\n"
          "Returns: List[cv2.DMatch]");

    // --- LUT‐based wedge filter --------------------------------------------
    m.def(
        "lut_epipolar_filter",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup, const Eigen::Matrix3f& F,
           float radius) {
            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);
            return lut_epipolar_filter(kp1, kp2, F, radius);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("radius"),
        "Lookup‐table epipolar wedge filter; returns candidates for each kp1.");

    // --- LUT‐based wedge filter Combined --------------------------------------------
    m.def(
        "lut_epipolar_filter_combined",
        [](const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           py::array_t<float>& F, const FilterConfig& filter_config, float radius) {
            auto t__cand_start = Clock::now();

            PointList kp1           = convert_tuple_list(kp1_tup);
            PointList kp2           = convert_tuple_list(kp2_tup);
            Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);
            auto candidates         = lut_epipolar_filter(kp1, kp2, F_eigen, radius);
            auto t_cand_end         = Clock::now();
            py::dict result;
            result["candidates"] = py::none();
            result["matches"]    = py::none();

            if (filter_config.type == FilterType::None) {
                result["candidates"] =
                    py::cast(candidates);  // No filtering, just return candidates
            } else if (filter_config.type == FilterType::GMSTest) {
                std::vector<cv::KeyPoint> kps1, kps2;
                kps1.reserve(kp1_tup.size());
                for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                kps2.reserve(kp2_tup.size());
                for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2,
                                                             candidates, filter_config.min_votes));
            } else if (filter_config.type == FilterType::RatioTest) {
                result["matches"] = py::cast(guided_match_ratio_test(
                    descriptors1, descriptors2, candidates, filter_config.ratio_threshold,
                    filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
            }
            auto t_end = Clock::now();
            if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
                // If we want to return all matches, we just return the candidates
                result["candidates"] = py::cast(candidates);
            }

            auto t_cand_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_cand_end - t__cand_start)
                    .count();
            auto t_filter_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_cand_end).count();
            result["cand_duration"]   = py::int_(t_cand_duration);
            result["filter_duration"] = py::int_(t_filter_duration);
            return result;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("F"), py::arg("filter_config"), py::arg("radius"),
        "Lookup‐table epipolar wedge filter; returns candidates for each kp1.");

    // --- Geometric distance filter -----------------------------------------
    m.def(
        "epipolar_geometric_distance_filter",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup, const Eigen::Matrix3f& F,
           float tol) {
            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);
            return epipolar_geometric_distance_filter(kp1, kp2, F, tol);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("tolerance"),
        "Brute‐force geometric distance epipolar filter.");

    m.def(
        "epipolar_geometric_distance_filter_combined",
        [](const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           py::array_t<float>& F, const FilterConfig& filter_config, float tol) {
            auto t__cand_start = Clock::now();

            PointList kp1           = convert_tuple_list(kp1_tup);
            PointList kp2           = convert_tuple_list(kp2_tup);
            Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);
            auto cands              = epipolar_geometric_distance_filter(kp1, kp2, F_eigen, tol);
            auto t_cand_end         = Clock::now();

            py::dict result;
            result["candidates"] = py::none();
            result["matches"]    = py::none();

            if (filter_config.type == FilterType::None) {
                result["candidates"] = py::cast(cands);  // No filtering, just return candidates
            } else if (filter_config.type == FilterType::GMSTest) {
                std::vector<cv::KeyPoint> kps1, kps2;
                kps1.reserve(kp1_tup.size());
                for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                kps2.reserve(kp2_tup.size());
                for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2,
                                                             cands, filter_config.min_votes));
            } else if (filter_config.type == FilterType::RatioTest) {
                result["matches"] = py::cast(guided_match_ratio_test(
                    descriptors1, descriptors2, cands, filter_config.ratio_threshold,
                    filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
            }
            auto t_end = Clock::now();

            if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
                // If we want to return all matches, we just return the candidates
                result["candidates"] = py::cast(cands);
            }

            auto t_cand_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_cand_end - t__cand_start)
                    .count();
            auto t_filter_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_cand_end).count();
            result["cand_duration"]   = py::int_(t_cand_duration);
            result["filter_duration"] = py::int_(t_filter_duration);

            return result;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("F"), py::arg("filter_config"), py::arg("tolerance"),
        "Brute‐force geometric distance epipolar filter.");

    // --- Guided GMS matcher (no ratio test, GMS outlier removal) -----------
    m.def(
        "guided_GMS_test",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           const std::vector<std::vector<int>>& candidate_list, int min_votes /*=6*/) {
            // 1) build cv::KeyPoint lists (size=1,fixed angle)
            std::vector<cv::KeyPoint> kps1, kps2;
            kps1.reserve(kp1_tup.size());
            for (auto& t : kp1_tup) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
            kps2.reserve(kp2_tup.size());
            for (auto& t : kp2_tup) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);

            // 2) call the C++ function
            return guided_GMS_test(kps1, kps2, descriptors1, descriptors2, candidate_list,
                                   min_votes);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("candidates"), py::arg("min_votes") = 6,
        "Guided matching with GMS outlier removal:\n"
        "  kp1, kp2         : list of (x,y) tuples of keypoints in image1 / image2\n"
        "  descriptors1,2   : (N×D) float32 descriptor arrays\n"
        "  candidates       : List[List[int]] of per–query candidate trainIdxs\n"
        "  min_votes        : GMS min‐votes threshold (default=6)\n"
        "Returns: List[cv2.DMatch] after best‐descriptor + GMS filtering.");

    // --- Epipolar hashing filter -------------------------------------------
    m.def(
        "epipolar_hash_filter_cpp",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup, const Eigen::Matrix3f& F, int bins,
           float tol, bool use_sampson, bool probe_adjacent, bool distance_filtering,
           std::optional<std::pair<int, int>> shape1, std::optional<std::pair<int, int>> shape2) {
            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);
            return epipolar_hash_filter_cpp(kp1, kp2, F, bins, tol, use_sampson, probe_adjacent,
                                            distance_filtering, shape1, shape2);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("bins") = 45, py::arg("tol") = 3.f,
        py::arg("use_sampson") = false, py::arg("probe_adjacent") = true,
        py::arg("distance_filtering") = true, py::arg("shape1") = py::none(),
        py::arg("shape2") = py::none(), "Angular‐hash epipolar filter.");

    m.def(
        "epipolar_hash_filter_cpp_combined",
        [](const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           py::array_t<float>& F, const FilterConfig& filter_config, int bins, float tol,
           bool use_sampson, bool probe_adjacent, bool distance_filtering,
           std::optional<std::pair<int, int>> shape1, std::optional<std::pair<int, int>> shape2) {
            auto t__cand_start = Clock::now();

            PointList kp1           = convert_tuple_list(kp1_tup);
            PointList kp2           = convert_tuple_list(kp2_tup);
            Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);
            auto cands =
                epipolar_hash_filter_cpp(kp1, kp2, F_eigen, bins, tol, use_sampson, probe_adjacent,
                                         distance_filtering, shape1, shape2);

            auto t_cand_end = Clock::now();

            py::dict result;
            result["candidates"] = py::none();
            result["matches"]    = py::none();

            if (filter_config.type == FilterType::None) {
                result["candidates"] = py::cast(cands);  // No filtering, just return candidates
            } else if (filter_config.type == FilterType::GMSTest) {
                std::vector<cv::KeyPoint> kps1, kps2;
                kps1.reserve(kp1_tup.size());
                for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                kps2.reserve(kp2_tup.size());
                for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2,
                                                             cands, filter_config.min_votes));
            } else if (filter_config.type == FilterType::RatioTest) {
                result["matches"] = py::cast(guided_match_ratio_test(
                    descriptors1, descriptors2, cands, filter_config.ratio_threshold,
                    filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
            }
            auto t_end = Clock::now();

            if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
                // If we want to return all matches, we just return the candidates
                result["candidates"] = py::cast(cands);
            }

            auto t_cand_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_cand_end - t__cand_start)
                    .count();
            auto t_filter_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_cand_end).count();
            result["cand_duration"]   = py::int_(t_cand_duration);
            result["filter_duration"] = py::int_(t_filter_duration);

            return result;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("F"), py::arg("filter_config"), py::arg("bins") = 45, py::arg("tol") = 3.f,
        py::arg("use_sampson") = false, py::arg("probe_adjacent") = true,
        py::arg("distance_filtering") = true, py::arg("shape1") = py::none(),
        py::arg("shape2") = py::none(), "Angular‐hash epipolar filter.");

    // --- Grid‐search epipolar filter ---------------------------------------
    m.def(
        "epipolar_grid_search_filter_cpp",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup, const Eigen::Matrix3f& F,
           float cell, float step, float tol, bool use_sampson, bool probe_adjacent,
           bool distance_filtering, std::optional<std::pair<int, int>> shape2) {
            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);
            return epipolar_grid_search_filter_cpp(kp1, kp2, F, cell, step, tol, use_sampson,
                                                   probe_adjacent, distance_filtering, shape2);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("cell") = 32.f,
        py::arg("step") = 20.f, py::arg("tol") = 3.f, py::arg("use_sampson") = false,
        py::arg("probe_adjacent") = true, py::arg("distance_filtering") = true,
        py::arg("shape2") = py::none(), "Staggered‐grid epipolar filter.");

    m.def(
        "epipolar_grid_search_filter_cpp_combined",
        [](const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           py::array_t<float>& F, const FilterConfig& filter_config, float cell, float step,
           float tol, bool use_sampson, bool probe_adjacent, bool distance_filtering,
           std::optional<std::pair<int, int>> shape2) {
            auto t__cand_start = Clock::now();

            PointList kp1           = convert_tuple_list(kp1_tup);
            PointList kp2           = convert_tuple_list(kp2_tup);
            Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);
            auto cands =
                epipolar_grid_search_filter_cpp(kp1, kp2, F_eigen, cell, step, tol, use_sampson,
                                                probe_adjacent, distance_filtering, shape2);

            auto t_cand_end = Clock::now();

            py::dict result;
            result["candidates"] = py::none();
            result["matches"]    = py::none();

            if (filter_config.type == FilterType::None) {
                result["candidates"] = py::cast(cands);  // No filtering, just return candidates
            } else if (filter_config.type == FilterType::GMSTest) {
                std::vector<cv::KeyPoint> kps1, kps2;
                kps1.reserve(kp1_tup.size());
                for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                kps2.reserve(kp2_tup.size());
                for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2,
                                                             cands, filter_config.min_votes));
            } else if (filter_config.type == FilterType::RatioTest) {
                result["matches"] = py::cast(guided_match_ratio_test(
                    descriptors1, descriptors2, cands, filter_config.ratio_threshold,
                    filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
            }
            auto t_end = Clock::now();

            if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
                // If we want to return all matches, we just return the candidates
                result["candidates"] = py::cast(cands);
            }

            auto t_cand_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_cand_end - t__cand_start)
                    .count();
            auto t_filter_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_cand_end).count();
            result["cand_duration"]   = py::int_(t_cand_duration);
            result["filter_duration"] = py::int_(t_filter_duration);

            return result;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("F"), py::arg("filter_config"), py::arg("cell") = 32.f, py::arg("step") = 20.f,
        py::arg("tol") = 3.f, py::arg("use_sampson") = false, py::arg("probe_adjacent") = true,
        py::arg("distance_filtering") = true, py::arg("shape2") = py::none(),
        "Staggered‐grid epipolar filter.");

    // --- Hybrid LUT‐binned filter ------------------------------------------
    m.def(
        "epipolar_lut_binned_filter_cpp",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup, const Eigen::Matrix3f& F,
           float tol_px, int bins, const std::string& edge_mode, bool use_sampson,
           std::optional<std::pair<int, int>> shape2, bool debug) {
            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);
            return epipolar_lut_binned_filter_cpp(kp1, kp2, F, tol_px, bins, edge_mode, use_sampson,
                                                  shape2, debug);
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("tol"), py::arg("bins") = 120,
        py::arg("edge_mode") = "geom", py::arg("use_sampson") = false,
        py::arg("shape2") = py::none(), py::arg("debug") = false,
        "Hybrid LUT‐binned epipolar filter.");

    m.def(
        "epipolar_lut_binned_filter_cpp_combined",
        [](const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           py::array_t<float>& F, const FilterConfig& filter_config, float tol_px, int bins,
           const std::string& edge_mode, bool use_sampson,
           std::optional<std::pair<int, int>> shape2, bool debug) {
            auto t__cand_start      = Clock::now();
            PointList kp1           = convert_tuple_list(kp1_tup);
            PointList kp2           = convert_tuple_list(kp2_tup);
            Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);

            auto cands = epipolar_lut_binned_filter_cpp(kp1, kp2, F_eigen, tol_px, bins, edge_mode,
                                                        use_sampson, shape2, debug);

            auto t_cand_end = Clock::now();

            py::dict result;
            result["candidates"] = py::none();
            result["matches"]    = py::none();

            if (filter_config.type == FilterType::None) {
                result["candidates"] = py::cast(cands);  // No filtering, just return candidates
            } else if (filter_config.type == FilterType::GMSTest) {
                std::vector<cv::KeyPoint> kps1, kps2;
                kps1.reserve(kp1_tup.size());
                for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                kps2.reserve(kp2_tup.size());
                for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2,
                                                             cands, filter_config.min_votes));
            } else if (filter_config.type == FilterType::RatioTest) {
                result["matches"] = py::cast(guided_match_ratio_test(
                    descriptors1, descriptors2, cands, filter_config.ratio_threshold,
                    filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
            }
            auto t_end = Clock::now();

            if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
                // If we want to return all matches, we just return the candidates
                result["candidates"] = py::cast(cands);
            }

            auto t_cand_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_cand_end - t__cand_start)
                    .count();
            auto t_filter_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_cand_end).count();
            result["cand_duration"]   = py::int_(t_cand_duration);
            result["filter_duration"] = py::int_(t_filter_duration);

            return result;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("F"), py::arg("filter_config"), py::arg("tol"), py::arg("bins") = 120,
        py::arg("edge_mode") = "geom", py::arg("use_sampson") = false,
        py::arg("shape2") = py::none(), py::arg("debug") = false,
        "Hybrid LUT‐binned epipolar filter.");

    // --- Wedge + segment‐tree filter ---------------------------------------
    m.def(
        "epipolar_wedge_filter_with_segment_tree",
        [](const std::vector<std::tuple<float, float>>& kp1_tup,
           const std::vector<std::tuple<float, float>>& kp2_tup, const Eigen::Matrix3f& F,
           float radius) {
            using Clock = std::chrono::high_resolution_clock;

            auto t0 = Clock::now();

            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);
            auto t1       = Clock::now();
            // std::cout << "[TIMER] Convert tuples to PointList: "
            //           << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
            //           t0).count()
            //           << " ms\n";
            auto return_data = epipolar_wedge_filter_with_segment_tree(kp1, kp2, F, radius);

            auto t2 = Clock::now();
            // std::cout << "[TIMER] Wedge + segment tree filter: "
            //           << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
            //           t1).count()
            //           << " ms\n";
            return return_data;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("F"), py::arg("radius"),
        "Epipolar‐wedge filter using a segment tree.");

    m.def(
        "epipolar_wedge_filter_with_segment_tree_combined",
        [](const py::array_t<float>& kp1_tup, const py::array_t<float>& kp2_tup,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors1,
           py::array_t<float, py::array::c_style | py::array::forcecast> descriptors2,
           py::array_t<float>& F, const FilterConfig& filter_config, float radius) {
            auto t__cand_start = Clock::now();

            PointList kp1 = convert_tuple_list(kp1_tup);
            PointList kp2 = convert_tuple_list(kp2_tup);

            Eigen::Matrix3f F_eigen = as_eigen_colmajor(F);
            auto cands = epipolar_wedge_filter_with_segment_tree(kp1, kp2, F_eigen, radius);

            auto t_cand_end = Clock::now();

            py::dict result;
            result["candidates"] = py::none();
            result["matches"]    = py::none();

            if (filter_config.type == FilterType::None) {
                result["candidates"] = py::cast(cands);  // No filtering, just return candidates
            } else if (filter_config.type == FilterType::GMSTest) {
                std::vector<cv::KeyPoint> kps1, kps2;
                kps1.reserve(kp1_tup.size());
                for (auto& t : kp1) kps1.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                kps2.reserve(kp2_tup.size());
                for (auto& t : kp2) kps2.emplace_back(std::get<0>(t), std::get<1>(t), 1.0f);
                result["matches"] = py::cast(guided_GMS_test(kps1, kps2, descriptors1, descriptors2,
                                                             cands, filter_config.min_votes));
            } else if (filter_config.type == FilterType::RatioTest) {
                result["matches"] = py::cast(guided_match_ratio_test(
                    descriptors1, descriptors2, cands, filter_config.ratio_threshold,
                    filter_config.use_adaptive_ratio, filter_config.apply_ratio_test));
            }
            auto t_end = Clock::now();

            if (filter_config.force_all_matches_return && filter_config.type != FilterType::None) {
                // If we want to return all matches, we just return the candidates
                result["candidates"] = py::cast(cands);
            }

            auto t_cand_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_cand_end - t__cand_start)
                    .count();
            auto t_filter_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_cand_end).count();
            result["cand_duration"]   = py::int_(t_cand_duration);
            result["filter_duration"] = py::int_(t_filter_duration);

            return result;
        },
        py::arg("kp1"), py::arg("kp2"), py::arg("descriptors1"), py::arg("descriptors2"),
        py::arg("F"), py::arg("filter_config"), py::arg("radius"),
        "Epipolar‐wedge filter using a segment tree.");

    // m.def("epipolar_wedge_filter_with_segment_tree_optimized",
    //       &epipolar_wedge_filter_with_segment_tree_optimized, py::arg("kp1"), py::arg("kp2"),
    //       py::arg("F"), py::arg("radius"), "Epipolar‐wedge filter using a segment tree
    //       optim.");

    m.def("BF_matching", &BF_matching, py::arg("descriptors1"), py::arg("descriptors2"),
          py::arg("ratio_thresh") = 0.75f, py::arg("use_adaptive_ratio") = false,
          py::arg("apply_ratio_test") = true,
          "Brute‐force matching of two sets of descriptors.\n"
          "Perform a BF matching between two sets of descriptors.\ns");

    m.def("all_matches_combined", &all_matches_combined, py::arg("kp1"), py::arg("kp2"),
          py::arg("descriptors1"), py::arg("descriptors2"), py::arg("points_3d_1"),
          py::arg("points_3d_2"), py::arg("F"), py::arg("filter_config"),
          py::arg("matching_config"), py::arg("thr_3d") = 0.1f,
          "Perform all matching strategies between two sets of descriptors.\n"
          "This includes BF matching, epipolar wedge filtering, and any other defined "
          "methods.\n");
}