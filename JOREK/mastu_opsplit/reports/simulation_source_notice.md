# Simulation runner source notice

`src/mastu_opsplit/runner.py` adapts bounded process handling and retry/failure classification from `tkit.datasets.jorek_run` at revision `14fcecb664cbb084b86051370bc415d8446e1637`. The original source is `/pitagora/home/userexternal/vgopakum/IntegratedModelling/tokamak-toolkit/src/tkit/datasets/jorek_run.py`. The adaptation avoids importing the toolkit JAX runtime and adds study-specific native output validation. Original license follows.

```text
MIT License

Copyright (c) 2026 Vignesh Gopakumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`patches/0001-model303-axisymmetric-perturbation.patch` is a proposed derivative of pinned JOREK source and remains LGPL-3.0-or-later. It is not part of, or permission to modify, the production executable.

`src/mastu_opsplit/restart.py` follows the cubic basis formulas in pinned JOREK `elements/mod_basisfunctions.f90` and the element-size/physical-gradient conventions in model303 `mod_elt_matrix_fft.f90`. The upstream JOREK source is LGPL-3.0-or-later. The basis parity test compiles the local pinned source in an isolated temporary directory; it does not edit production or the dependency checkout.

The isolated shared-attachment and finish-grid axis-initialization patches modify pinned JOREK grid utilities under the same LGPL-3.0-or-later terms. Their source diffs and linked derivative build provenance are retained in `patches/0002-grid-shared-patch-attachments.patch`, `patches/0003-finish-grid-initialize-axis-before-rtree.patch`, and the corresponding CPU build reports.
