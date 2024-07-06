# PS-Poly

### Description

The PS-Poly algorithm is a particle detection program designed to separate individual features based on shape using images taken by atomic force microscopy. The program sorts particles into the following groups: linear, looped, branched without looping, branched with looping, overlapped, and noise. For linear molecules, persistence length is calculated using the worm-like chain model. Due to interpolation that is performed to increase the pixel density of the image, the persistence length result is achieved with subpixel precision.

Code is open source and available in both Igor Pro (WaveMetrics, Inc.) and Python. You will find tutorials for both versions in the tutorials/ directory of the GitHub repository. Source code is available in the pspoly/ directory.

### Igor Package Requirements

Igor Pro 7

### Python Package Requirements

Python >= 3.8

NumPy >= 1.24.4

Matplotlib >= 3.7.5

SciPy >= 1.10.1

scikit-image >= 0.21.0