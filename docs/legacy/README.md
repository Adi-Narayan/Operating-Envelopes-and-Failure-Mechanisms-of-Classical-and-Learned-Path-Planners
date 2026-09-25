# Legacy documents

These documents were written during development. They predate the corrected evaluation reported in the paper and are kept for the record only. Where they disagree with the paper or the code, the paper and `experiments/` are authoritative. Known discrepancies are listed in `experiments/FINDINGS.md`. They include:

- **`APF_Methodology.docx`** describes the parameter sweep on an earlier 10 × 10 world with θ = 15°, step 0.12 and goal tolerance 0.20. Its k_r grid (0.01–0.50) does not contain the k_r = 0.85 used in the paper.
- **`2D_ARC_Methodology.docx`** has several errors against the code:
  - it describes the nine ray sensors as spread over 360° at 40° spacing (the code uses the forward 180° at 22.5°);
  - it calls the action penalty quadratic (it is linear);
  - it makes the LOS reward conditional on an unobstructed line of sight (it is given every step);
  - it describes one update per episode (the code makes one per environment step).
- **`ARC Metrics.pdf`** describes the separate notebook-style 2D environment (`ARCEnvironment`: 2-D action, progress-shaped reward), not the `UAVEnvironment` agent evaluated in the paper.
