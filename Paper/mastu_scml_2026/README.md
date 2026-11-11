# MAST-U videos and SCML presentation

The committed [portable package](MASTU_SCML_2026_presentation_assets.zip) contains the finished videos, updated PDF, insert slides and gallery. Extract it into any directory and open `index.html`; no scratch data or Python environment is needed for viewing. The original source deck is tracked alongside the build scripts.

Presentation assets for `SCML_2026_NOs4POs.pdf`. The original PDF stays unchanged.
Large outputs live behind [artifacts](artifacts), a local link to
`/pitagora_scratch/userexternal/vgopakum/mastu_comparison_videos_20261111`.
The final PDF contains stills; the H.264 MP4 files are separate, portable movie assets.


- [Updated 30-page presentation](artifacts/presentation/SCML_2026_NOs4POs_with_MASTU.pdf)
- [Seven equation/result insert slides](artifacts/presentation/MASTU_implicit_OpsSplit_insert.pdf)
- [Video gallery](artifacts/index.html) and [portable presentation package](artifacts/MASTU_SCML_2026_presentation_assets.zip)
- Density movies: [case 004](artifacts/videos/mastu_004/mastu_004_density_change.mp4), [case 010](artifacts/videos/mastu_010/mastu_010_density_change.mp4)

The comparison uses the first two existing held-out cases (`mastu_004`, `mastu_010`),
seed 17, identical initial states, native mesh, observation times and training-only
normalization. AR and Neural ODE use their final 2,000-update GPU checkpoints.
Implicit Neural OpsSplit uses the final 64-update learned correction with the
corrected autonomous native source implementation. No model is retrained or
selected by these video results. This is a two-case presentation experiment.

The native equations are

\[
M_*\dot q=F_{\mathrm{native}}(q)+D_\theta(q),\qquad
D_\theta(q)=\mathcal A[G_\theta(q)-G_\theta(q_*)].
\]

The deployed IMEX Euler evolution step is

\[
(M_*-hL_*)\Delta q=h[F_{\mathrm{native}}(q_n)+D_\theta(q_n)],\qquad
q_{n+1}=\mathcal P(q_n+\Delta q).
\]

The native current/vorticity constraints and boundary rows are solved together.
`P` denotes the physical guards and Mach-wall enforcement. The temporal operator
and tangent are frozen; nonlinear native physics and the learned defect remain
explicit. The anchor is the native reference restart, not the training mean.
The known reference forcing remains nonzero. Most of the accuracy gain comes
from retaining native nonlinear physics; the learned correction adds about 5.16%
over its matched full-physics control on these two cases.

GPU rollout timings synchronize before and after autonomous inference, include
deployed guards, and exclude setup, scoring, export and rendering. The primary
numbers average two case medians, each with three warm repeats. Native JOREK
timing is historical wall time on four CPU MPI ranks, including native I/O and
checks. It is not a hardware-matched GPU benchmark. Historical optimizer-update
times are recorded separately in `artifacts/summary.json`; they are not fresh
training benchmarks.

The video RMS labels are instantaneous, per-field native-volume RMS divided
by the training field scale. The presentation MSE averages all seven fields over
frames 1:H and then over cases; these are deliberately different summaries.

The primary 96-interval window spans 1.405913 microseconds and is 24 times the
four-interval training window. Reference data use 384 native integration steps;
the main implicit comparison uses 192 microsteps. Step-refinement and H384
experiments retain the same weights. Ground truth ends at H96. Later forecasts
can establish survival or failure under their respective gates, not accuracy.
Stopped forecasts and missing reference frames are blank, never frozen or filled.

Reproduction uses these repository scripts:

- `JOREK/mastu_opsplit/scripts/benchmark_method_videos.py`: one CUDA method/case worker; saves timings, native coefficients and physical audits.
- `JOREK/mastu_opsplit/scripts/assemble_method_videos.py`: checks pinned checkpoints, source, normalization, native reference arrays and matched prefixes; assembles video inputs and summary.
- `JOREK/mastu_opsplit/scripts/render_method_videos.py`: native bicubic R–Z rendering, shared fixed scales, native volume errors and verified MP4 frame counts.
- `Paper/mastu_scml_2026/build_slides.py`: equation/result slides and a separate revised PDF, preserving all original pages.

The exact executed protocol, follow-up declaration, commands, raw reports and
source hashes are in `artifacts`. Task-local `pypdf` and `imageio-ffmpeg` packages
are in `artifacts/dependencies`; the existing research Python environment was
not changed. Rendered fields use native JOREK simulation units, without an
unverified conversion to SI density or temperature.

Measured primary results (mean of case medians):

| Method | GPU rollout / 96 intervals | Mean normalized MSE | Updates | Historical update time |
|---|---:|---:|---:|---:|
| Autoregressive | 6.21 s | 25.1636 | 2,000 | 699.46 s |
| Neural ODE | 6.21 s | 1.44588 | 2,000 | 690.98 s |
| Implicit Neural OpsSplit | 47.68 s | 0.00187614 | 64 | 335.37 s |

Native reference wall times: 4,294.57 s (`mastu_004`) and 4,261.68 s (`mastu_010`), on four CPU MPI ranks. Historical optimizer-update times exclude preparation/checkpoint/evaluation and came from jobs with possible GPU contention; they are descriptive, not a matched training-speed comparison.

The exploratory one-microstep variant averages **24.93 s** and **0.00235238 MSE**, 25.4% more error than two microsteps for 1.91× lower rollout time. Case 004 changes by only +0.33% MSE, but case 010 changes by +43.1%; two microsteps remain the primary video configuration.

All three H384 trajectories reach the requested horizon under their respective guards. Minimum positivity limiter factors are AR `2.47e-14`, Neural ODE `0.00584`, implicit `0.517`; continued output depends on substantial guard intervention. No native ground truth exists beyond H96.
