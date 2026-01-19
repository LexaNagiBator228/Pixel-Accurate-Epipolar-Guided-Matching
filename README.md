# Pixel-Accurate Epipolar Guided Matching

<div align="center">

**[3DV 2026]**

[📄 Paper (Coming Soon)](#) | [💻 Code (Coming Soon)](#) | [🌐 Project Page](https://yourname.github.io/Pixel-Accurate-Epipolar-Guided-Matching/)

---

*Official implementation of "Pixel-Accurate Epipolar Guided Matching" accepted at 3DV 2026*

</div>

## 🔥 News
- **[Jan 2026]** Paper accepted to 3DV 2026! 🎉
- **[Coming Soon]** Code and pre-trained models will be released
- **[Coming Soon]** Paper will be available on arXiv

## 📖 Abstract

Keypoint matching can be slow and unreliable in challenging conditions such as repetitive textures or wide-baseline views. In such cases, known geometric relations (e.g., the fundamental matrix) can be used to restrict potential correspondences to a narrow epipolar envelope, thereby reducing the search space and improving robustness. These epipolar-guided matching approaches have proved effective in tasks such as SfM; however, most rely on coarse spatial binning, which introduces approximation errors, requires costly post-processing, and may miss valid correspondences. We address these limitations with an exact formulation that performs candidate selection directly in angular space using tolerance circles and segment trees for logarithmic-time queries.

## 🏗️ Method Overview

Our approach introduces:
- **Angular Interval Formulation**: Reformulates epipolar envelope constraints as 1D angular intervals viewed from the epipole
- **Tolerance Circle Method**: Associates each keypoint with tolerance circles that define angular visibility regions 
- **Segment Tree Data Structure**: Enables logarithmic-time candidate queries using balanced segment trees
- **Exact Geometric Filtering**: Guarantees pixel-level precision without approximation errors
- **Per-Keypoint Control**: Supports individual tolerance settings for enhanced flexibility

## 🎯 Key Features

- ✅ **Exact formulation** without approximation errors
- ✅ **Logarithmic-time queries** using segment tree data structure  
- ✅ **Perfect candidate recall** - recovers all valid correspondences
- ✅ **Scalable performance** up to 50k keypoints per image
- ✅ **Per-keypoint control** with pixel-level tolerance settings

## 📊 Results

### Quantitative Results
*Coming soon - detailed benchmark results will be added*

### Qualitative Results  
*Coming soon - visual comparisons will be added*

## 🚀 Getting Started

### Requirements
*Coming soon - will be updated when code is released*

### Installation
```bash
# Installation instructions will be provided when code is released
git clone https://github.com/yourname/Pixel-Accurate-Epipolar-Guided-Matching.git
cd Pixel-Accurate-Epipolar-Guided-Matching
# pip install -r requirements.txt
```

### Quick Demo
*Coming soon - demo instructions will be provided*

## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{nasypanyi2026pixel,
    title={Pixel-Accurate Epipolar Guided Matching},
    author={Oleksii Nasypanyi and Rameau Francois Bernard Julien},
    booktitle={International Conference on 3D Vision (3DV)},
    year={2026}
}
```

## 🙏 Acknowledgements

We thank the reviewers for their constructive feedback and the 3DV 2026 organizing committee.

## 📧 Contact

For questions about this work, please contact: [your-email@domain.com](mailto:your-email@domain.com)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>
