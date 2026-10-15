# Verification log: "Operating Envelopes of Classical and Learned Path Planners"

Running log of what was checked, what was found, and the edit each finding needs in the paper.
Numbers marked **[FINAL]** come from the re-run on the new machine and can go into the paper.
All runs have finished; earlier [PENDING] markers in Section 1 are resolved by Section 3.

- Paper checked: `Apf_research.pdf` (PDF only; no .tex was available, so edits are given as replacement text).
- Code: `github.com/Adi-Narayan/Operating-Envelopes-and-Failure-Mechanisms-of-Classical-and-Learned-Path-Planners` (cloned into `Research-Paper/`), plus `models.zip` (the original 2D ARC checkpoints).
- New machine for all re-runs: AMD Ryzen 7 5800HS (8 cores / 16 threads), 16 GB DDR4 (15.4 GB usable), NVIDIA RTX 3050 Laptop GPU (4 GB), Windows 11, Python 3.12.7, NumPy 2.0.2, PyTorch 2.5.0, Shapely 2.1.2.
- Scripts: `experiments/*.py`. Raw outputs: `experiments/out/*.json`.

---

## 0. Protocol changes made before re-running (these change every number in the paper)

The code did not implement the "like-for-like" comparison the paper describes. The following was fixed in the code, and all tables were re-measured under it:

| # | Problem in the original code | Fix |
|---|---|---|
| P1 | **The ARC "custom environment" is a different layout from APF's.** ARC trained and tested on `env_apf_urban`, a 10×10 notebook layout scaled ×50. Every non-outlier obstacle sits somewhere else than in APF's Corridor Gauntlet (for example, rectangle B1 is at (150,300) in ARC but (180,280) in APF). The paper's "identical in both codebases" is false. | Added `corridor_gauntlet` to `arc_2d/code/config.py`: an exact copy of the APF geometry (symmetric-difference area 0.0 for all 10 obstacles). ARC was retrained on it. |
| P2 | Goal radius differed: APF 18 m, ARC 10 m. | Both use 10 m (ARC's trained `d_min_target`). |
| P3 | APF appended a final segment from the stopping point to the exact goal centre, and counted it as a step. | Removed. The path ends at the first point inside the goal radius, for both planners. |
| P4 | APF never recorded a collision. After 5 failed escapes it teleported out of the obstacle and kept going, so "0 collisions" was true by construction. ARC's collisions used forward-only ray sensors. | One criterion for both: exact clearance to any obstacle boundary < 5 m, checked after every step (agent into obstacle or obstacle into agent). ARC's native sensor collision is also kept; it implies clearance < 5 m anyway. |
| P5 | Dynamic seeds were time-based (`int(time.time()*1000)`), so runs were not reproducible. | Seeds are fixed. Environment `s` = obstacle-velocity seed `s` (0–49). The APF planner seed is `1000+s`. |
| P6 | The paired 50-environment harness is not in the repository at all. | Rebuilt as `experiments/harness.py` + `paired_dynamic.py`: one obstacle timeline per seed, integrated on a 1/30 s grid, so both planners see the identical motion. |
| P7 | Agent speeds relative to the obstacles differed silently. APF moves 1.8 m per 0.1 s (18 m/s). ARC moves 3.0 m per 1.0 s (3 m/s). Obstacles move at up to 4.75 m/s, i.e. faster than ARC but much slower than APF. | Both planners move at 18 m/s in the paired suite (ARC's 3.0 m step spans 1/6 s). |
| P8 | ARC evaluation start was random: `ARCTester` draws one random start in [30,100]² and a random initial heading per run. d_straight therefore differs from 608.1 m. | ARC is evaluated from (35,35) with the initial heading pointing at the goal, using the deterministic policy. Run-to-run spread for ARC comes from independently trained seeds. |
| P9 | Travel budgets differed: APF 6000 × 1.8 m = 10.8 km, ARC 500 × 3 m = 1.5 km. | Both get 10.8 km (APF 6000 steps, ARC 3600 steps). |

---

## 1. Findings on the problems you listed

### 1.1 Table 4 Panel B is broken — CONFIRMED (cause identified)
- `arc_2d/code/test.py::test_multiple_runs` averages path length and efficiency over **all** runs, collided runs included. The original model on its own layout gives "efficiency" 103.9 ± 15.6% under that code: impossible above 100%, and it happens because short collided paths are averaged in. That is the same pooled-over-all-runs bug. A 353.11 m mean path (< 608.1 m straight line) can only arise this way.
- The mismatches you flagged are all real: 608.1/353.11 = 172% (not 81.21%), and 608.1/851.50 = 71.4% (not 65.8%).
- The harness that produced Table 4 is not in the repo, so the original numbers cannot be reproduced. **Panel A and Panel B are re-measured from scratch** restricting path metrics to the paired shared-success set. **→ new Table 4: §3.2.**

### 1.2 Panel B text 94.8% vs table 81.21% — CONFIRMED
Both numbers are unsupported. The 94.8% equals the static Table 3 ARC figure exactly, so it looks copied across. Replace with the re-measured value: ARC 93.0 ± 5.6% vs APF 71.3 ± 21.7% on 58 shared pairs (§3.2).

### 1.3 Static ARC evaluation contradicts itself (7.1 vs 7.3) — CONFIRMED, cause identified
What the code actually did (`test.py::ARCTester`):
- The start position is drawn **once** at random from [30,100]² (not (35,35)).
- The initial heading is drawn **uniformly at random on every run** (`UAVEnvironment.reset`).
- The policy is deterministic (no exploration noise).

So neither description in the paper is right. The ±17.8 m spread came from the random initial heading. d_straight was measured from the random start, not 608.1 m.
Evidence: the original `arc_best.pth` under the original tester gives start (68.4, 80.1), d_straight = 552.7 m, and 7/10 successes on its own training layout (not 10/10).
Fix: see P8. Rewrite 7.1/7.3 to say: "ARC is evaluated with its deterministic policy from q0 = (35,35) with the initial heading pointing at the goal; the reported spread is across N independently trained policies."

### 1.4 ARC SD does not follow from the step SD — CONFIRMED
In the ARC environment every step is exactly 3.0 m. Verified: L/steps = 3.000 on every run. So SD(L) must be 3.0 × SD(steps) = 3.0 × 5.6 = 16.8 m, not 17.8 m. The efficiency SD would be ≈ 94.8 × 16.8/641.1 ≈ 2.5, not 2.6. Superseded by the re-measured Table 3 (§3.1).

### 1.5 Abstract says θ "selected by parameter sweeps" — CONFIRMED
θ is fixed at 25° everywhere in the code, and the paper says so in 2.1, 3.3 and 7.4. The methodology docx's sweep even held θ fixed (at 15°, on the 10×10 world). Also:
- The k_r grid in the docx (0.01–0.50) does **not contain** the reported k_r = 0.85.
- Table 6 says "Optimised (k_a, k_r) per layout", but `static_apf.py` uses the same (5.0, 0.85) on all five layouts.
- **New:** only the ratio k_r/k_a affects the trajectory (the force is normalised and the bias is negligible). (5.0, 0.85), (2.5, 0.425) and (1.0, 0.17) give paths that differ by at most 1.4 mm [FINAL]. The 15×15 grid is therefore effectively a 1-D sweep over k_r/k_a. Say so in 3.7/6.3.
- **Sweep result [FINAL]** (`gain_sweep.py`, custom env, θ = 25°, score S of Eq. 9). **(5.0, 0.85) is not the sweep's argmin.**
  - Extended grid (k_a ∈ [1,5], k_r ∈ [0.15,1.55]): 177/225 succeed, 48 time out. The best S = 736.22 (L = 736.2 m) occurs at k_r/k_a ≈ 0.25–0.27, e.g. (3.57, 0.95). The reported (5.0, 0.85), ratio 0.17, has S = 741.66 (L = 741.6 m) and ranks **88th of 177**.
  - Docx grid (k_r ∈ [0.01,0.50]): 65/225 succeed, 153 time out, 7 collide; the best S is again 736.22 at a ratio of 0.255.
  - The sweep-optimal ratio does **not** transfer: on the other layouts (seed 0), ratio 0.255 gives Dense Forest 75.1% (vs 86.0%), Wall Maze 77.5% (vs 87.3%) and Scattered 76.4% (vs 77.8%), for +0.6 points on the custom env. (5.0, 0.85) is the better single gain pair across layouts.
- **Recommended wording (author decision):** the gains were *not* selected as a sweep argmin. Describe them honestly: "A single gain pair (k_a, k_r) = (5.0, 0.85), i.e. k_r/k_a = 0.17, is used on every layout. A 15 × 15 sweep on the custom environment finds a slightly shorter path at k_r/k_a ≈ 0.26 (736.2 m vs 741.6 m), but that ratio lowers efficiency by 9–11 points on Dense Forest and Wall Maze, so we keep the cross-layout pair."

Abstract fix: "... with a single fixed gain pair used on every layout and its rotation angle held fixed at 25°". Do not say "selected by parameter sweeps" for either the gains or θ.

### 1.6 Obstacle count ("ten-obstacle configuration" of 3+3+3) — CONFIRMED
The Corridor Gauntlet has **10** obstacles: 2 rectangles (60 × 150 m), 2 squares (60 × 60 m), 3 triangles (circumradius 32.5 m), 3 circles (one r = 25 m, and two r = 32.5 m circles at (70,430) and (430,70)). The two corner circles are the "outliers": they stay stationary in the dynamic suite (`outlier: True` in `dynamic_apf/apf.py`).
Replacement for 6.1: "a ten-obstacle configuration of four axis-aligned rectangles (two 60 × 150 m blocks and two 60 × 60 m squares), three triangles and three circles. Two of the circles, placed near the (70, 430) and (430, 70) corners, are designated outliers and remain stationary in the dynamic suite." Also fix the Fig. 1 caption, which says "three circles, three axis-aligned rectangles, three triangles".

### 1.7 3D section compares against a 2D dynamic efficiency the paper no longer reports — CONFIRMED
Rewrite 7.6 to compare with the 2D numbers that are actually reported: 2D static APF (Table 3) and the 2D dynamic APF mean over the 50-run suite (Table 4). New 3D values are in 2.3 below.

### 1.8 Section 2.3 claims not supported by citations — CONFIRMED
[18,19] are the original RRT/RRT* papers. They contain no comparison with RL agents. The APF-beats-value-based-RL claim has no citation at all.

Fix: delete both specific claims. Support the paragraph's actual point (controlled comparisons are rare) with a verified source:
- X. Xiao, B. Liu, G. Warnell, P. Stone, "Motion planning and control for mobile robot navigation using machine learning: a survey," *Autonomous Robots*, vol. 46, no. 5, pp. 569–597, 2022 (verified via Crossref, DOI 10.1007/s10514-022-10039-8). Its abstract states: "to date, there has not been much direct comparison between the classical and emerging paradigms to this problem."

Suggested replacement for the first two sentences of 2.3:
> "Controlled comparisons between classical and learned planners are rarer than the area's confidence suggests; a recent survey of learning-based navigation notes that there has been little direct comparison between the two paradigms [Xiao et al. 2022]."

Keep [18,19] only if RRT is mentioned for another reason. Otherwise drop them from the bibliography.
(Xu et al., "Benchmarking RL techniques for autonomous navigation", ICRA 2023, arXiv:2210.04839, exists, but its abstract makes no classical-vs-RL claim, so do not cite it for one.)

### 1.9 Abstract `\ref{sec:github}` renders "Section 9" — CONFIRMED (from the PDF)
The abstract says "link is given at Section 9", but Section 9 is the Conclusion. The Data Availability section is unnumbered, and it prints the paper title instead of a URL. Fix: drop the cross-reference from the abstract ("The code and evaluation logs are publicly available; see Data Availability."), and put the URL `https://github.com/Adi-Narayan/Operating-Envelopes-and-Failure-Mechanisms-of-Classical-and-Learned-Path-Planners` in the Data Availability section.

### 1.10 "Escape deflections make L exceed s × steps" (marked VERIFY) — REFUTED
Under the original protocol there were **zero** escape steps and **zero** push-outs on all five 2D layouts and on the 3D static run [FINAL]. The excess comes from two things:
1. the final segment from the stopping point (within 18 m) to the goal centre, which was appended to the path, and
2. that segment being counted as a step (`steps = len(path) − 1`).

| Run (original protocol) | steps | L (m) | appended segment (m) | escapes |
|---|---|---|---|---|
| Custom env (10 seeds) | 408.6 ± 0.5 | 750.60 ± 0.003 | 16.9 | 0 |
| Dense Forest | 390 | 716.70 | 16.5 | 0 |
| Wall Maze | 384 | 705.61 | 16.2 | 0 |
| Scattered Mixed | 430 | 789.71 | 17.5 | 0 |
| Narrow Passage | 6000 | 10800.0 | – (timeout) | 0 |
| 3D static | 449 | 824.16 | 17.8 | 0 |

These reproduce the paper's Table 3/6/7 APF numbers exactly, which confirms the mechanism. That is why 2D and 3D show the same ~1.835 ratio: the same appended segment of up to 18 m. Under the corrected protocol, L = 1.8 × steps exactly (max deviation 5e−11 m). Delete the escape explanation from 3.5, the Table 6 footnote, and the text.

### 1.11 225 sweep runs "in roughly fifteen minutes" — PARTLY CONFIRMED (the time comes from timeouts)
A successful static run takes about 0.25–0.55 s, so 225 successful runs would take 1–2 min. The measured sweep times (single process, with three ARC trainers running concurrently on the same CPU):
- docx grid: **26.7 min**, of which 153 of 225 combinations hit the 6000-step cap (about 8 s each);
- extended grid: **8.8 min**, 48 timeouts.

"Roughly fifteen minutes" is only plausible because most of the budget goes into timed-out combinations. Rewrite 6.3: "A 15 × 15 = 225-combination sweep took 9–27 min of wall time depending on the grid, dominated by combinations that exhaust the 6000-step budget." Also drop "per layout": the sweep was only run on the custom environment.

### 1.12 The 0.8 m clamp can never activate — CONFIRMED
Across all static runs on all five layouts (original and corrected protocol), the smallest raw distance ever passed to the repulsive term was **15.3 m**, and the clamp fired **0** times [FINAL]. In the dynamic runs, the corrected protocol terminates at clearance < 5 m, so it cannot fire there either. Rewrite 3.3: "the distance is clamped below at 0.8 m as a numerical guard; with the 6 m collision margin enforced it was never active in any reported run."

### 1.13 "Quantised in steps of 0.2" applies to rates, not the reward — CONFIRMED, plus a bigger problem
- The per-episode reward is the mean of the UAVs' continuous returns. 99% of the logged reward values are not multiples of 0.2. Only the success and collision *rates* are quantised.
- **Bigger problem:** the TensorBoard log that was in the repo (`arc_2d/Tensorboard_data`, host `Jen-2.local`, a Mac; removed in the cleanup, still in git history) does **not** match the paper's training run:

| Claim in paper | Repo TensorBoard log |
|---|---|
| 300 episodes | 200 episodes |
| 5 UAVs (quantised in 0.2) | 3 UAVs: rates are k/3 ∈ {0, 0.333, 0.667, 1} |
| 21.45 min wall clock | 7.3 min |
| reward rises from ≈ −50 | early rewards ≈ −86 to −115 |
| Table 5: Q = 16.94, actor loss = −17.08, success 0.54, collision 0.16 | final Q 17.1, actor loss −17.3; smoothed success 0.62–0.85, collision 0.09–0.27 depending on smoothing |

So Fig. 5 and Table 5 cannot be reproduced from the repo. Both are replaced by the retraining on the new machine (§3.5, `out/figures/fig5_arc_training.png`).
- `models.zip` (checkpoints ep50–ep300, saved 18:50→19:15) is consistent with a 300-episode run of about 25 min. `arc_best.pth` is byte-for-byte the same policy as `arc_checkpoint_ep300.pth` (identical evaluation results).

### 1.14 Triangles move "up to 3.8 units/step" vs APF's 1.8 m/step — REFUTED (units and ranges in 6.2 are wrong)
Obstacle velocities in the code are in **m per simulated second**. Positions advance by `vel × dt` with dt = 0.1 s per APF step. The ranges printed in 6.2 also omit the `BASE_SPEED_SCALE = 1.25` factor, and the rectangle lower bound is wrong. The correct ranges (uniform on [0.7 v_max, v_max]):

| Type | Paper says | Code (m/s) | Max displacement per APF step (dt = 0.1 s) |
|---|---|---|---|
| rectangle | [0.875, 1.750] | [1.225, 1.750] | 0.175 m |
| square | [1.680, 2.400] | [2.100, 3.000] | 0.300 m |
| circle (non-outlier) | [2.030, 2.900] | [2.5375, 3.625] | 0.363 m |
| triangle | [2.660, 3.800] | [3.325, 4.750] | 0.475 m |
| 3D sphere | – | [2.5375, 3.625] | 0.363 m |
| 3D box | – | [1.575, 2.250] | 0.225 m |

The APF agent moves 1.8 m per 0.1 s = 18 m/s, so every obstacle is at least 3.8× slower than the agent. That answers the reviewer question. The paper should state speeds in m/s and state the agent's speed.

Also undocumented in 6.2, and it must be disclosed:
- **Goal protection:** any obstacle whose centre comes within 50 m of the goal is pushed away (`GOAL_PROTECTION_RADIUS`, both 2D and 3D).
- Inter-obstacle contacts use AABB overlap. Velocities are exchanged **along** the contact normal (not "normal to the contact axis"), and each speed is then rescaled to its pre-contact value.
- Walls: the obstacle is shifted back inside and the relevant velocity component is reflected.

---

## 2. Other claims checked

### 2.1 Static 2D APF under the corrected protocol — [FINAL]
10 seeded runs per layout (bias seeds 0–9), k_a = 5.0, k_r = 0.85, θ = 25°, same gains on every layout. The initial bias b does not change the outcome or the step count of any run (path lengths agree to < 1e−10 m), but it moves the trajectories by up to 2.8 cm (see §4.15).

| Layout | obstacles | success | steps | L (m) | η (%) | escapes / push-outs | min clearance (m) |
|---|---|---|---|---|---|---|---|
| Custom (Corridor Gauntlet) | 10 | 10/10 | 412 | 741.6 | 82.0 | 0 / 0 | 24.1 |
| Dense Forest | 12 circles | 10/10 | 393 | 707.4 | 86.0 | 0 / 0 | 15.3 |
| Wall Maze | 8 (5 walls, 2 squares, 1 triangle) | 10/10 | 387 | 696.6 | 87.3 | 0 / 0 | 20.5 |
| Scattered Mixed | 11 | 10/10 | 434 | 781.2 | 77.8 | 0 / 0 | 24.0 |
| Narrow Passage | 11 | 0/10 (timeout) | 6000 | 10800 | 5.6 | 0 / 0 | 29.5 |

Mean efficiency over the four solved layouts: (82.0+86.0+87.3+77.8)/4 = **83.3%** (paper: 82.3%).
Text fix: Wall Maze is "8 obstacles (5 wall segments, 2 squares, 1 triangle)", not "8 rectangular barriers".

### 2.2 Rotation-angle sensitivity (Sec. 7.4, Fig. 6, Sec. 8.2) — claims partly REFUTED [FINAL]
θ swept 0–90° in 5° steps, seed 0, corrected protocol. S = success; T = timeout; the number is efficiency (%).

```
Corridor Gauntlet  0:S84.9 5:S84.2 10:S83.4 15:S80.6 20:S81.4 25:S82.0 30:S82.4 35:S82.8 40:S83.0 45:S83.4 50:S83.4 55:S83.6 60:S83.6 65:S83.6 70:S83.6 75:S83.2 80:S82.4 85:S80.4 90:S74.2
Dense Forest       0:S85.7 5:S89.4 10:S89.6 15:S89.6 20:S85.5 25:S86.0 30:S86.2 35:S78.6 40:S79.5 45:S79.9 50:S80.1 55:S80.2 60:S80.4 65:S80.4 70:S80.1 75:S91.3 80:S91.3 85:S90.3 90:S85.1
Wall Maze          0:T 5.6 5:T 5.6 10:T 5.6 15:S86.2 20:S86.8 25:S87.3 30:S87.8 35:S88.0 40:S88.4 45:S88.7 50:S88.9 55:T 5.6 60:S73.6 65:S73.4 70:S73.0 75:S71.9 80:S70.2 85:S67.4 90:S57.9
Scattered Mixed    0:S83.2 5:S83.8 10:S77.0 15:S77.3 20:S77.7 25:S77.8 30:S78.2 35:S78.4 40:S78.8 45:S78.9 50:S79.1 55:S79.1 60:S87.5 65:S88.7 70:S89.1 75:S89.4 80:S89.4 85:S88.7 90:S83.8
Narrow Passage     0:T 5.6 ... 45:T 5.6 50:S66.6 55:S66.6 60:S66.7 65:S66.6 70:S66.8 75:S66.9 80:S66.8 85:S66.0 90:S60.8
```

- "Larger angles produce visibly tighter cornering and shorter paths" on the custom env: **false.** The best efficiency is at θ = 0° (84.9%), and 90° is the worst (74.2%). The curve is non-monotonic, with a dip at 15–20°.
- "Smooth and predictable in θ, with no sharp failure across the range": true on the custom env only. On other layouts there are jumps (Dense Forest 30°→35°, Scattered 55°→60°) and outright failures (Wall Maze at 0–10° and at 55°).
- Sec. 8.2 "the angle that would break [Narrow Passage] is closer to 45°": close, but not 45°. It fails at every θ ≤ 45° and succeeds at every θ ≥ 50° (66.6–66.9%).
- Sec. 8.2 "at 45° ... weakening obstacle avoidance on the other layouts": **not supported**. At 45° all four other layouts still succeed. **Every θ in {50°, 60°, 65°, …, 90°} clears all five layouts, including Narrow Passage**; the only failure at θ ≥ 50° is Wall Maze at 55°. θ = 25° is therefore not the only reasonable fixed choice, and a single fixed θ (e.g. 50°) solves every layout, including the one the paper presents as the characteristic failure. The paper must acknowledge this. Suggested framing: θ is held at 25° by design (chosen before the sweep, not tuned); the sweep shows that a larger fixed angle clears Narrow Passage, so that failure is a property of θ = 25° rather than an inherent limit of fixed-handedness rotation. The efficiency cost of θ = 50° versus 25° is +1.4 points on the custom env, −5.9 on Dense Forest, +1.6 on Wall Maze and +1.3 on Scattered Mixed.
- Pure radial APF (θ = 0) solves the custom env. The motivating claim that rotation is needed there is not supported on this layout; it is needed on Wall Maze.

### 2.3 3D APF (Table 7) — [FINAL]
Corrected protocol, 10 seeds each.

| | success | steps | L (m) | η (%) | escapes | min clearance |
|---|---|---|---|---|---|---|
| 3D static | 10/10 | 453.0 ± 0.0 | 815.4 ± 0.0 | 91.3 ± 0.0 | 0 | 38.1 m |
| 3D dynamic | 10/10 | 455.6 ± 18.4 | 820.1 ± 33.1 | 90.9 ± 3.4 | 0 | 23.9 m |

The 3D static layout matches the paper (4 boxes + 6 spheres, 2 stationary outlier spheres). Idle wall times: 232 ± 33 ms (static), 397 ± 15 ms (dynamic) (§3.6).

### 2.4 ARC observation / reward / sensors (Sec. 4) — text does not match the code
- **Sensors:** 9 rays over the **forward 180°** at **22.5°** spacing (`(i−4)·π/8` relative to heading), not "40° intervals around the agent". The methodology docx's "full 360°" is also wrong.
- **Observation:** [ψ = UAV heading (world frame), χ = world-frame azimuth from UAV to target, d0/500, 9 ranges/50 m]. Not "relative bearing" and "heading error".
- **LOS reward:** r_LOS = λ(π/2 − |χ − ψ|), given **every step**. It is **not** conditional on the line of sight being unobstructed.
- **Obstacle term:** σ·‖Σ_i (1/d_i − 1/R) e_i‖², with e_i the unit vector of sensor i's angular offset.
- **Action penalty:** α·|Δψ|, i.e. **linear**, not quadratic.
- **Terminal:** −100 on collision **or leaving the arena**, +100 on reaching the goal.
- **Training:** one gradient update per environment step once the buffer holds 32 samples. The five exploring-start UAVs run **sequentially** within an episode, not in parallel.

### 2.5 Hardware / provenance claims
- The paper says all experiments ran on an i5-13500H / 8 GB laptop. The repo shows the 2D ARC TensorBoard log came from a Mac (`Jen-2.local`, with macOS `powermetrics` hooks in `test.py`). `arc_3d.py` has Kaggle/Colab paths (`/kaggle/working/...`, `_in_colab()`). Unless the authors confirm otherwise, the "same laptop" and "all on consumer hardware" claims are not supported for the ARC work. All 2D numbers in the revised paper come from the Ryzen 7 5800HS machine above. The 3D ARC section (qualitative) should state where it was run.
- `arc_3d.py` in the repo is the **scaled 500³ version** (WORLD_SCALE = 25, Stage 2 = 26 obstacles incl. decorative swirls, goal radius 20 m). Section 5 describes a 20 × 20 × 15 m arena with 23 obstacles and a 0.8 m goal radius, and that version is not in the repo. Section 5 is qualitative, but its numbers (23 obstacles, 6 m/12 m curriculum spheres) cannot be checked from the code.

### 2.6 Minor text issues found
- 3.4 triangle kernel: inside a triangle the code's gradient points from the nearest boundary point toward q, i.e. *inward*, and there is no "nearest edge midpoint". Inside an AABB the gradient is zero. None of this ever happens in the reported runs (min clearance ≥ 15 m), but the text should describe it correctly or drop the inside case.
- 3.5: after five failed escapes the agent is pushed out along the nearest obstacle's normal to 3 m beyond the margin; the run is **not** "recorded as a failure". Each escape attempt is a 3.24 m (1.8 s) perpendicular step plus N(0, 0.15²) noise.

---

### 2.7 Compute device for 2D ARC
A 2D ARC update (batch 32, LSTM 128, T = 5) costs 20.2 ms on one CPU thread (14.5 ms with 4 threads) against 30.8 ms on the RTX 3050 (CUDA 12.1). For this network size, kernel-launch and synchronisation overhead dominates, so all 2D ARC training ran on the CPU, one thread per seed, three seeds concurrently (about 300 MB RAM each). Paper 6.3 should say the 2D ARC was trained on the CPU. The GPU is not used by any 2D result.

### 2.8 Where the paper's Table 3 ARC column came from — reproduced [FINAL]
`urban_head_to_head.py`: the original `arc_best.pth` on its own training layout (`env_apf_urban`), start (35,35), corrected protocol.

| | success | steps | L (m) | η (%) |
|---|---|---|---|---|
| ARC, heading toward goal (deterministic) | 1/1 | 212 | 636.0 | 95.6 |
| ARC, random initial heading (original tester), 10 runs | **4/10** (4 collisions, 2 out of bounds) | 214.8 ± 3.9 (successes) | 644.3 ± 11.6 | 94.4 ± 1.7 |
| APF on the same layout, 10 runs | 10/10 | 406 | 730.8 | 83.2 |
| *Paper Table 3, ARC* | *10/10* | *213.7 ± 5.6* | *641.1 ± 17.8* | *94.8 ± 2.6* |

The paper's ARC path metrics match **success-only averages on ARC's own layout under random initial headings**. The "10/10 success" does not reproduce (4/10). The ARC trajectory passes within 5.75 m of an obstacle, only 0.75 m above the collision threshold. The original model on the actual Corridor Gauntlet collides (2/10 under the original tester; collision at step 120 from (35,35) with heading toward the goal).

### 2.9 Paired 50-environment suite, Option A (ARC's layout, original model), both planners at 18 m/s [FINAL]
`evaluate.py --layout urban --models models_original/arc_best.pth --tag urban_original`

| Panel A (all 50) | APF | ARC |
|---|---|---|
| Success | 48/50 (96%) | 30/50 (60%) |
| Collision | 2/50 (obstacle moved into the agent ~78 m from the goal) | 20/50 |
| Timeout / out of bounds | 0 / 0 | 0 / 0 |

| Panel B (28 environments both solve) | APF | ARC |
|---|---|---|
| Path length (m) | 799.6 ± 200.4 | 620.8 ± 15.1 |
| Efficiency (%) | 79.7 ± 15.7 | 98.0 ± 2.3 |
| Steps | 444.2 ± 111.3 | 206.9 ± 5.0 |
| More efficient in | 2/28 | 26/28 |

- ARC solves 2 environments APF does not (seeds 12, 13). The paper's "strict subset" claim does **not** hold here.
- APF over all 50: min clearance 4.16 m (a collision), 0 escapes, 0 push-outs.
- This differs strongly from the paper's 18% vs 98%. The relative speed of agent and obstacles is decisive; see the 3 m/s sensitivity in 2.10.

### 2.10 Speed-ratio sensitivity — the paper's dynamic result is a speed confound [FINAL]
Same 50 environments, same layout (Option A), same original model, but both planners at 3 m/s (ARC's native speed: 3 m per 1 s step; APF 1.8 m per 0.6 s). Obstacles (up to 4.75 m/s) are then up to 1.6× faster than the agent. `--speed 3 --tag urban_original_v3`.

| Ground speed (both planners) | APF success | APF collisions | ARC success | ARC collisions | Shared successes | η on shared (APF / ARC) |
|---|---|---|---|---|---|---|
| 18 m/s | 48/50 | 2 | 30/50 | 20 | 28 | 79.7% / 98.0% |
| 3 m/s | 25/50 | 25 | 6/50 | 44 | 2 | 93.8% / 98.6% |
| *Paper Table 4* | *49/50* | *0* | *9/50* | *40* | *9* | – |

At 3 m/s APF also needed 50 escape steps and 49 push-outs (0 at 18 m/s). ARC solves 4 environments APF fails (1, 27, 47, 48).

**Interpretation.** The paper's pair (APF 98%, ARC 18%) matches APF running at 18 m/s and ARC at 3 m/s. In the original code, APF moves 1.8 m per 0.1 s and ARC 3 m per 1 s, against the same obstacle speeds in m/s. At equal speed the gap narrows and both planners degrade together as obstacles get faster. APF's robustness to motion is largely a consequence of the agent being ≥ 3.8× faster than every obstacle.
**Paper consequence.** Sections 7.2, 8.1, the abstract and the conclusion must (i) state the agent and obstacle speeds, (ii) run both planners at the same speed, and (iii) report the speed-ratio sensitivity as part of the operating envelope. It is itself an envelope result: APF's envelope ends when obstacles approach the agent's speed.

## DECISION (authors, resolved): the shared environment is the Corridor Gauntlet (Option B)
ARC is retrained on the Corridor Gauntlet (5 seeds). ARC's original layout (`env_apf_urban`) and the original model are kept as the second layout of the transfer matrix and as supplementary evidence (2.8–2.10, 2.14). Final numbers: Section 3.

### 2.12 APF inside ARC's 3D arena (APF versatility) [FINAL]
`apf_in_arc3d.py`. The arena, SDF kernels and wind model are exec'd verbatim from `arc_3d/arc_3d.py`: 500³ m, start (25,25,25), goal (462.5,462.5,300), d_straight = 677.1 m. ARC's own success (≤ 20 m) and collision (SDF < 1.75 m) definitions apply. The APF planner is unchanged (k_a 5.0, k_r 0.85, θ 25° about z, s 1.8 m); only the distance kernel is swapped for ARC's exact sphere/box/cylinder SDFs. The 3 "swirl" zones are visual only in ARC's code (no physics) and are ignored, as ARC ignores them. 10 seeds per cell.

| ARC curriculum stage | solid obstacles | wind | success | steps | L (m) | η (%) | min clearance (m) | escapes / push-outs |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 | off | 10/10 | 391.0 ± 0.0 | 703.8 | 96.2 | 31.6 | 0 / 0 |
| 0 | 4 | on | 10/10 | 390.7 ± 1.1 | 704.9 ± 1.1 | 96.1 ± 0.1 | 31.3 | 0 / 0 |
| 1 | 15 | off | 10/10 | 438.0 ± 0.0 | 788.4 | 85.9 | 19.8 | 0 / 0 |
| 1 | 15 | on | 10/10 | 437.6 ± 1.6 | 789.1 ± 2.4 | 85.8 ± 0.3 | 19.7 | 0 / 0 |
| 2 (full) | 23 (10 spheres, 8 boxes, 5 cylinders) | off | 10/10 | 487.6 ± 0.5 | 877.7 ± 0.9 | 77.1 ± 0.1 | 31.8 | 0 / 0 |
| 2 (full) | 23 | on | 10/10 | 484.1 ± 14.3 | 872.9 ± 26.1 | 77.6 ± 2.3 | 27.0 | 0 / 0 |

APF solves every stage of the arena the 3D ARC was built for, with no training, the same gains and θ as in 2D, and only a cylinder kernel added. The 0.8 m clamp never fired. ARC's own Stage-2 budget is 1200 steps × ≤ 3.75 m; APF's paths (≤ 878 m) fit easily. **No 3D ARC checkpoint is available** (the code points at `/kaggle/working/...`), so the ARC side of a 3D head-to-head cannot be evaluated here. If the authors have the 3D checkpoints, `apf_in_arc3d.py` gives APF numbers under exactly ARC's metric.

Suggested paper use: replace the Section 7.6 sentence "We do not report a 3D APF-vs-ARC head-to-head ..." with the table above as a Table 7b ("APF in the 3D ARC arena"), stated as an APF-only versatility result. Keep the statement that no stable large-arena 3D ARC number is reported.

### 2.13 Corrected APF 3D results in the paper's own 3D environment (Table 7) [FINAL]
See 2.3. 3D static: 10/10, 453 steps, 815.4 m, η = 91.3%. 3D dynamic: 10/10, 455.6 ± 18.4 steps, 820.1 ± 33.1 m, η = 90.9 ± 3.4%. 0 escapes, min clearance 38.1 m (static) and 23.9 m (dynamic). Together with 2.12, APF runs unchanged in two independently designed 3D worlds, 2D static and 2D dynamic.

### 2.14 Speed sweep on ARC's layout with the original model [FINAL]
`speed_sweep.py --layout urban --models models_original/arc_best.pth`. 50 environments per speed; ratio r = v_obs,max / v_agent with v_obs,max = 4.75 m/s.

| agent speed (m/s) | r | APF success | ARC success |
|---|---|---|---|
| 3.0 | 1.58 | 25/50 | 6/50 |
| 4.0 | 1.19 | 44/50 | 10/50 |
| 4.75 | 1.00 | 45/50 | 12/50 |
| 6.0 | 0.79 | 47/50 | 15/50 |
| 9.0 | 0.53 | 47/50 | 8/50 |
| 12.0 | 0.40 | 46/50 | 9/50 |
| 18.0 | 0.26 | 48/50 | 30/50 |
| 27.0 | 0.18 | 50/50 | 26/50 |

APF degrades smoothly and substantially only when obstacles are faster than the agent (r > 1). ARC succeeds far less often than APF at every ratio, and does best at 18–27 m/s, where the obstacles barely move during an episode. Between r = 0.4 and 1.2 it stays at 8–15/50 without a clear trend (one model; the 95% Wilson half-width at n = 50 is about ±13 points, i.e. ±6–7 successes). This is consistent with a static-trained policy that has no model of obstacle motion. The corridor version with the retrained seeds is the main figure; this urban one is supplementary.

### 2.15 ARC policy quality oscillates across checkpoints; "best" is chosen by training reward [FINAL]
`eval_checkpoints.py`: static layout, start (35,35). "det" = heading toward the goal; "rand" = successes over 10 seeded random initial headings (the training start distribution).

| model (layout it was trained on) | ep50 | ep100 | ep150 | ep200 | ep250 | ep300 | arc_best |
|---|---|---|---|---|---|---|---|
| corridor seed 0: det | OOB | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| corridor seed 0: rand | 0/10 | 7/10 | 8/10 | 0/10 | 0/10 | 3/10 | 3/10 |
| corridor seed 2: det | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| corridor seed 2: rand | 0/10 | 0/10 | 8/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| corridor seed 3: det | OOB | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| corridor seed 3: rand | 0/10 | 0/10 | 0/10 | 1/10 | 1/10 | 6/10 | 0/10 |
| corridor seed 4: det | OOB | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ (233 st, 699 m) |
| corridor seed 4: rand | 0/10 | 0/10 | 0/10 | 5/10 | 0/10 | 0/10 | 5/10 |
| original (urban): det | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| original (urban): rand | 10/10 | 6/10 | 9/10 | 0/10 | 8/10 | 4/10 | 4/10 |

(✓ success, ✗ collision, OOB out of bounds)

- The policy learns the layout (7–8/10 around ep100–150) and then **collapses** (0/10 at ep200–250), partially recovering by ep300. The original run shows the same pattern. This is the known instability of DDPG with a single critic, fixed exploration noise and a 4000-transition FIFO buffer that turns over every few episodes. It is the same pathology the paper's Section 5 uses to motivate TD3 in 3D, and it is already present in 2D.
- Where a corridor checkpoint succeeds deterministically, its path is 684–732 m (η = 83–89%), against APF's 741.6 m (82.0%). On the Corridor Gauntlet, ARC's in-distribution efficiency advantage is therefore at most a few points, not the 94.8% vs 81.0% of the original Table 3 (which came from the other layout; see 2.8).
- `arc_best.pth` is selected by the rolling 20-episode *training reward*. It is not the best checkpoint by evaluation, and for both retrained seeds it fails the deterministic test.
- Paper consequence: (i) state the model-selection rule; (ii) report ARC at arc_best (the pipeline default, primary), at ep300 (final), and at the best checkpoint by evaluation, labelled as an optimistic upper bound because it is selected on the test layout; (iii) report the oscillation itself. It strengthens the setup-cost argument: a usable policy exists only in part of the 300-episode run, and picking it needs an evaluation step that APF does not need.

### 2.11 Run-log note: power throttling
Between roughly 01:20 and 02:05 on 25 Sep the laptop ran on battery with the Balanced plan. The CPU dropped to about 1.9 GHz and the ARC training rate fell from 20–30 s/episode to 100–240 s/episode (episode lengths were unchanged at ~120–150 steps). Wall-clock training times for corridor seeds 0–4 therefore include that period. Training cost is reported from the per-episode rate on AC power, not raw elapsed time. All timing measurements (timing.py) are taken on AC power with the machine otherwise idle.

## 3. Final results (corrected protocol, all on the Ryzen 7 5800HS machine) [FINAL]

Full tables with every model variant: `out/report.md` (generated by `report.py` from `out/final_*.json`, `out/speed_sweep_corridor_*.json`, `out/timing.json`, `out/apf3d.json`, `out/apf_in_arc3d.json`, `out/verify_apf_static.json`). Figures: `out/figures/`.

ARC variants per seed: **best** = `arc_best.pth`, the pipeline's own choice by rolling training reward (primary, pre-declared). **final** = ep300. **oracle** = the checkpoint with most successes on its own static layout (optimistic upper bound, selected on the evaluation layout).

### 3.1 New Table 3 — static head-to-head, Corridor Gauntlet
APF: 10 runs (bias seeds 0–9). ARC: 5 independently trained seeds, each evaluated deterministically from (35,35) with the heading toward the goal. d_straight = 608.1 m.

| | APF | ARC best (primary) | ARC final (ep300) | ARC oracle (upper bound) |
|---|---|---|---|---|
| Success | 10/10 | **1/5 seeds** | 1/5 | 4/5 |
| Steps | 412.0 ± 0.0 | 233 | 232 | 236.8 ± 6.3 |
| Path length (m) | 741.6 ± 0.0 | 699.0 | 696.0 | 710.3 ± 18.9 |
| Efficiency (%) | 82.0 ± 0.0 | 87.0 | 87.4 | 85.7 ± 2.3 |
| Step displacement (m) | 1.8 | 3.0 | 3.0 | 3.0 |
| Steps per 100 m | 55.6 | 33.3 | 33.3 | 33.3 |
| Per-step planner cost (ms) | 0.435 | 1.506 ± 0.021 | – | – |
| Compute per 100 m (ms) | **24.2** | **50.2** | – | – |
| Run wall time (ms) | 320 ± 13 | 423 ± 11 (successful runs) | – | – |
| Training | none | 300 episodes, 112–139 min per seed (median 122 min; one non-learning seed 353 min) | | |

ARC best, per seed: s0 collision at step 74; s1 timeout (never learned, circles for 10.8 km); s2 collision at step 126; s3 collision at step 75; s4 success.
Per-step cost: APF = force evaluation + 6 m margin check; ARC = 9 Shapely ray casts + LSTM/actor forward pass. Both were measured by replaying a recorded path on the idle machine. ARC's per-step cost is **3.5×** APF's, and per 100 m it is **2.1×** (the paper said 2× per step and 1.2× per 100 m).

### 3.2 New Table 4 — paired 50-environment dynamic suite, Corridor Gauntlet, both planners at 18 m/s
**Panel A (all 50 environments).** APF: **50/50** (95% Wilson CI 93–100%), 0 collisions, 0 timeouts, min clearance 5.66 m, 1 escape step and 1 push-out over all runs.

| ARC (5 seeds) | success per seed | pooled success | collisions | timeouts | out of bounds |
|---|---|---|---|---|---|
| best (primary) | 26, 0, 1, 26, 5 → **11.6 ± 13.3 / 50** | 58/250 (23%) | 185 | 2 | 5 |
| final | 26, 0, 0, 19, 2 → 9.4 ± 12.2 | 47/250 | – | – | – |
| oracle | 25, 0, 21, 19, 28 → 18.6 ± 11.0 | 93/250 | 147 | 2 | 8 |

- **No ARC model, in any variant, solves an environment APF fails.** The "strict subset" statement holds on the Corridor. Exact McNemar test APF vs ARC per model: p ≤ 1.2 × 10⁻⁷ for every seed that succeeds at all.

**Panel B (paired, only environments both solve; pooled over seeds).**

| | APF | ARC best | | APF | ARC oracle |
|---|---|---|---|---|---|
| shared (seed, environment) pairs | 58 | 58 | | 93 | 93 |
| Path length (m) | 1155.0 ± 1275.3 | 656.3 ± 43.5 | | 1155.7 ± 1343.1 | 688.9 ± 55.0 |
| Efficiency (%) | 71.3 ± 21.7 | **93.0 ± 5.6** | | 71.9 ± 20.8 | 88.9 ± 7.3 |
| Steps | 643.2 | 218.8 | | 642.6 | 229.6 |
| ARC more efficient | | 53/58 | | | – |
| Wilcoxon signed-rank p | | 2 × 10⁻¹⁰ | | | 5 × 10⁻¹⁴ |

APF's path length over all 50 environments is 1093.1 ± 1246.2 m, and its efficiency 72.7 ± 19.6% (range 7.2–99.1%). The spread comes from a few environments where moving obstacles force long detours.

### 3.3 Transfer matrix (ARC best / oracle, 5 seeds per training layout, both planners at 18 m/s)

| trained on → evaluated on | static det. success, best / oracle | dynamic success /50, best | dynamic /50, oracle | APF static / dynamic |
|---|---|---|---|---|
| Corridor → Corridor | 1/5 / 4/5 | 11.6 ± 13.3 | 18.6 ± 11.0 | 10/10 / 50/50 |
| Corridor → urban | 2/5 / 1/5 | 10.2 ± 10.4 | 10.8 ± 9.7 | 10/10 / 48/50 |
| urban → Corridor | 1/5 / 0/5 | 16.0 ± 10.0 | 31.2 ± 4.0 | 10/10 / 50/50 |
| urban → urban | 2/5 / 5/5 | 12.0 ± 6.9 | 26.2 ± 9.7 | 10/10 / 48/50 |

- The seed-to-seed spread (SD ≈ 7–13 environments) is as large as any layout effect, so in-distribution vs transfer is not resolvable at 5 seeds with arc_best.
- The one clear in-distribution effect is on the *static* test with the oracle checkpoint: 4/5 and 5/5 in-distribution, against 1/5 and 0/5 when transferred.
- The urban layout is easier for ARC to learn (training success over the last 50 episodes 0.59–0.66, vs 0–0.34 on the Corridor; see 3.5).

### 3.4 Speed-ratio envelope, Corridor Gauntlet (Fig. `fig_speed_envelope.png`)
Both planners at the same speed v; r = 4.75 m/s / v; 50 identical seeded environments per speed; ARC = arc_best of the 5 corridor seeds.

| v (m/s) | r | APF | ARC per seed | ARC mean ± SD |
|---|---|---|---|---|
| 3 | 1.58 | 30/50 | 3, 0, 1, 8, 4 | 3.2 ± 3.1 |
| 4 | 1.19 | 37/50 | 14, 0, 3, 8, 9 | 6.8 ± 5.4 |
| 4.75 | 1.00 | 47/50 | 19, 0, 3, 10, 8 | 8.0 ± 7.3 |
| 6 | 0.79 | 46/50 | 24, 0, 12, 13, 4 | 10.6 ± 9.3 |
| 9 | 0.53 | 48/50 | 25, 0, 7, 20, 3 | 11.0 ± 10.9 |
| 12 | 0.40 | 46/50 | 20, 0, 7, 36, 5 | 13.6 ± 14.5 |
| 18 | 0.26 | 50/50 | 26, 0, 1, 26, 5 | 11.6 ± 13.3 |
| 27 | 0.18 | 49/50 | 32, 0, 0, 14, 12 | 11.6 ± 13.1 |

- APF holds 92–100% for r ≤ 1 and falls to 74% at r = 1.19 and 60% at r = 1.58. **Its envelope edge is where obstacles become faster than the agent.**
- ARC is below APF at every ratio. It degrades toward r > 1 as well (8.0 → 3.2 mean successes), but seed variance dominates across the whole range.

### 3.5 New Table 5 / Fig. 5 — ARC training (5 seeds per layout, 300 episodes, 5 exploring-start UAVs per episode)
Last-50-episode means from `train_summary.json` (training rollouts, with exploration noise):

| seed | Corridor: success / collision / reward | urban: success / collision / reward |
|---|---|---|
| 0 | 0.12 / 0.73 / 68.5 | 0.66 / 0.28 / 237.6 |
| 1 | 0.00 / 0.00 / −14.7 (never learned; episodes run to the 500-step limit) | 0.60 / 0.33 / 223.1 |
| 2 | 0.18 / 0.74 / 80.5 | 0.59 / 0.39 / 216.1 |
| 3 | 0.26 / 0.63 / 85.7 | 0.62 / 0.34 / 221.7 |
| 4 | 0.34 / 0.54 / 117.0 | 0.66 / 0.30 / 239.3 |

Wall clock: 112–139 min per seed with 3–6 seeds sharing the 8-core CPU (1 thread each); corridor seed 1 took 353 min. Rates are quantised in steps of 0.2 (5 UAVs); the reward is not. Fig. 5 is regenerated from the 5 corridor seeds.
Checkpoint oscillation: see 2.15 (all 10 retrained seeds + the original model). Every seed that learns shows collapse-and-recovery between checkpoints.

### 3.6 3D (APF only)
Paper's 3D environment (Table 7): static 10/10, 453 steps, 815.4 m, 91.3%, 232 ± 33 ms per run (0.51 ms/step). Dynamic 10/10, 455.6 ± 18.4 steps, 820.1 ± 33.1 m, 90.9 ± 3.4%, 397 ± 15 ms per run (0.87 ms/step).
ARC's 3D arena (Table 7b, 2.12): 60/60 across stages 0–2, calm and wind. Idle-machine run times: 101 ms (stage 0), 520 ms (stage 1), 887 ms (stage 2). The finite-difference SDF gradients over 23 obstacles dominate.

### 3.7 What changes in the paper's story
1. **Static head-to-head.** The paper said ARC is clearly better in-distribution (94.8% vs 81.0%, 10/10). On the shared layout, ARC's primary model succeeds for **1 of 5 seeds**. Where an ARC policy succeeds, it is only 3–5 points more efficient (85.7–87.4% vs 82.0%). ARC's per-100 m compute is 2.1× APF's, not 1.2×.
2. **Dynamic suite.** The paper said 98% vs 18%. At equal speed it is **100% vs 23% pooled (0–52% per seed)**. The strict-subset statement holds. ARC's efficiency advantage on shared successes is real and significant (93.0% vs 71.3%, p = 2 × 10⁻¹⁰).
3. **New envelope result.** Success vs obstacle-to-agent speed ratio. APF's robustness ends at r ≈ 1, which the original unequal-speed setup hid.
4. **New setup-cost evidence.** Training outcome depends heavily on the seed (1 of 5 never learns). Policy quality oscillates between checkpoints. The pipeline's reward-based `arc_best` choice is usually not the policy that evaluates best.
5. **3D.** APF runs unchanged in two independent 3D worlds, including every stage of the arena the 3D ARC was built for. The 3D ARC checkpoints are not available, so no 3D head-to-head is claimed.
6. **Future work.** Envelope-informed hybrid switching (see PAPER_EDITS §8.4).

---

## 4. Additional experiments for the RAS submission [FINAL; written into the paper]

Scope agreed with the authors. Everything runs on the laptop; test 4 (ARC at 1000 episodes) was dropped.

| Test | What it answers | Status |
|---|---|---|
| 1. APF with ARC's constraints, factorial over sensing {full, range-limited 50 m, ARC's 9 forward rays} × turn-rate cap {off, 54°/s = ARC's 9° per 1/6 s} × step/decision rate {1.8 m @ 10 Hz, 3.0 m @ 6 Hz} (`apf_variants.py`, `test1_matched.py`) | Does APF's robustness survive when sensing, turn rate and replanning rate are matched to ARC? | running |
| 3a. TD3 trained **with moving obstacles** (same motion model and time scale as the evaluation suite, disjoint seeds), 10 seeds (`dyn_train_env.py`, `train_arc_seed.py --algo td3 --dynamic`) | Removes the "static-trained policy of course fails on moving obstacles" objection | training |
| 3b. TD3 on the static layout, 10 seeds (`--algo td3`) | Is the 1-of-5 in-distribution result an artefact of the DDPG baseline? | queued |
| 3c. SAC on the static layout, 5 seeds (optional, last in the queue) | Second modern baseline | queued |
| 2. Sensor and actuation noise, both planners | Robustness axis of the envelope | to do |
| 5. 100 randomly generated layouts, both planners | Replaces the single-layout limitation | to do |

The all-switches-off APF variant reproduces `harness.run_apf` exactly (max path difference 0.0 m on the static layout and three dynamic seeds).
First observation, static Corridor Gauntlet (deterministic): with only ARC's 9 forward rays, APF fails. Without the turn cap it oscillates in the gap between the central triangle and the eastern block until timeout. With the cap it detects obstacles too late to turn away at 54°/s and clips them (clearance 4.2–4.5 m). With full sensing, the cap and ARC's step rate cost less than 0.5 efficiency points (82.0% → 81.7–81.8%).

**The discussion will be written only after the numbers are in.** Tests 1 and 3 may move APF below 100% on the dynamic suite, or put a dynamic-trained TD3 above it.

### 4.1 Test 1 results [FINAL; in the paper as Table matched]
`test1_matched.py` → `out/test1_matched.json`. Both planners at 18 m/s. The dynamic suite is the 50 paired Corridor Gauntlet environments of Table 4. Static layouts give one deterministic run each.

**Dynamic suite, APF success out of 50** (every non-success is a collision; there were no timeouts):

| sensing | turn cap | 10 Hz, 1.8 m | 6 Hz, 3.0 m | η on successes (10 Hz / 6 Hz) |
|---|---|---|---|---|
| full | off | 50 | 50 | 72.7 / 70.7 |
| full | on (54°/s) | 40 | 40 | 81.6 / 81.4 |
| 50 m range, exact geometry | off | 48 | 49 | 68.6 / 66.3 |
| 50 m range, exact geometry | on | 32 | 32 | 82.4 / 82.2 |
| ARC's 9 forward rays | off | 35 | 37 | 73.8 / 73.2 |
| ARC's 9 forward rays | on | 35 | **36** (all three matched) | 80.3 / 81.5 |

For reference: the corridor-trained ARC (arc_best, 5 seeds) scores 11.6 ± 13.3 / 50 on the same environments (Table 4).

**Static layouts** (succ/coll/timeout, η %):

| variant | Corridor | Dense Forest | Wall Maze | Scattered | Narrow Passage |
|---|---|---|---|---|---|
| full, no cap (both rates) | succ 82 | succ 86 | succ 87 | succ 78 | timeout |
| full, cap (both rates) | succ 82 | succ 85 | succ 87 | succ 78 | timeout |
| range, no cap | timeout | succ 84–87 | succ 85–86 | succ 85–86 | timeout |
| range, cap @10 Hz / @6 Hz | coll / coll | coll / succ 89 | succ 85 | succ 85 | timeout |
| rays, no cap | timeout | succ 69–71 | succ 71–72 | succ 71–73 | timeout |
| rays, cap (both rates) | coll | succ 85 | succ 85 | succ 84 | coll |

Measured facts only:
- The step/decision rate changes dynamic success by ≤ 2 environments in every sensing/cap cell.
- The turn cap alone (full sensing) removes 10 of 50 dynamic successes.
- Restricting sensing to ARC's rays alone removes 13–15.
- With all three asymmetries removed, APF succeeds in 36/50.
- On the static Corridor Gauntlet, every sensing-limited variant fails, and the fully matched APF collides.

### 4.2 Test 5 results: 100 random layouts (APF + the 5 corridor-trained ARC seeds) [FINAL; TD3/SAC in §4.8]
`robustness_suite.py --test layouts` → `out/layouts_apf_arc.json`. Layouts are from `random_layouts.py` (the Corridor Gauntlet primitive mix at random positions, ≥ 25 m gaps). Static = one deterministic run per layout. Dynamic = one motion seed per layout, 18 m/s.

| planner | static success /100 (coll / timeout / OOB) | η static | dynamic success /100 (coll / timeout / OOB) | η dynamic |
|---|---|---|---|---|
| APF, paper (full, no cap, 10 Hz) | 68 (0 / 32 / 0) | 87.9 | 98 (2 / 0 / 0) | 74.3 |
| APF, all ARC constraints (rays, cap, 6 Hz) | 90 (4 / 6 / 0) | 87.4 | 80 (20 / 0 / 0) | 87.3 |
| ARC seed 0 | 84 (11 / 0 / 5) | 92.5 | 58 (34 / 0 / 8) | 92.1 |
| ARC seed 1 | 0 (0 / 100 / 0) | – | 0 (93 / 7 / 0) | – |
| ARC seed 2 | 12 (83 / 0 / 5) | 94.9 | 6 (88 / 0 / 6) | 95.1 |
| ARC seed 3 | 60 (39 / 0 / 1) | 94.3 | 42 (54 / 0 / 4) | 94.6 |
| ARC seed 4 | 34 (62 / 0 / 4) | 90.7 | 21 (75 / 0 / 4) | 88.8 |

**The paper's APF gets trapped on 32/100 static layouts, and all 32 are solvable** (another planner reaches the goal on each one). It stalls in open space, 180–555 m from the goal (median 438 m), oscillating between two points one step apart. At the stall point in layout 2, |F_att| = 2168, |F_rep| = 2209 and |F_total| = 47: attraction and summed repulsion nearly cancel. Restricting repulsion to obstacles within 50 m does not remove the traps (60/100 static). A 12-variant factorial on the 100 static layouts is running to isolate the factor responsible (`out/layouts_static_factorial.json`).

### 4.3 Test 2 results: noise (APF + the 5 corridor-trained ARC seeds) [FINAL; TD3/SAC in §4.8]
`robustness_suite.py --test noise` → `out/noise_apf_arc.json`. The 50 paired dynamic environments at 18 m/s; noise = (SD of perceived obstacle distance, SD of executed heading). Success out of 50:

| level | APF paper | APF all-ARC-constraints | ARC s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|---|
| L0 (0 m, 0°) | 50 | 36 | 26 | 0 | 1 | 26 | 5 |
| L1 (1 m, 2°) | 48 | 32 | 27 | 0 | 1 | 22 | 7 |
| L2 (2.5 m, 5°) | 48 | 37 | 21 | 0 | 2 | 22 | 13 |
| L3 (5 m, 10°) | 47 | 29 | 15 | 0 | 2 | 16 | 7 |

L0 reproduces Table 4 and Test 1 exactly (consistency check of the noise code path).

### 4.4 Which factor removes APF's free-space traps on random layouts [FINAL]
`out/layouts_static_factorial.json`. All 12 APF variants on the 100 random static layouts (one deterministic run each). Success / timeout / collision:

| sensing | no cap, 10 Hz | no cap, 6 Hz | cap, 10 Hz | cap, 6 Hz |
|---|---|---|---|---|
| full | 68 / 32 / 0 | 69 / 31 / 0 | 81 / 13 / 6 | 83 / 11 / 6 |
| 50 m range | 60 / 40 / 0 | 62 / 38 / 0 | 80 / 13 / 7 | 80 / 12 / 8 |
| ARC's 9 rays | 68 / 30 / 2 | 68 / 29 / 3 | 89 / 3 / 8 | 90 / 6 / 4 |

Measured:
- **The turn-rate cap is the factor that removes most traps.** Timeouts fall from 29–40 to 3–13 in every sensing condition, while collisions rise from 0–3 to 4–8 (corrected: earlier text said '4–8 extra collisions'; for rays at 6 Hz the increase is 1).
- Ray sensing alone does not change the success count (68 → 68); a 50 m range limit alone lowers it (60–62).
- The step/decision rate changes results by ≤ 2 layouts.
- Mechanism (from 4.2): the unconstrained planner ping-pongs between two points one step apart at a point where attraction and summed repulsion nearly cancel. A heading-rate limit makes that reversal impossible.

### 4.5 TD3 checkpoint evaluation and training collapse [FINAL; all runs in out/collapse.json]
`eval_checkpoints.py --layout corridor` → `out/checkpoints_corridor_td3dyn04.json`, `out/checkpoints_corridor_td304.json` (35 checkpoints each: ep50…ep300 + best). Static Corridor Gauntlet, fixed start; "rand" = successes over the 10 seeded random initial headings. Two of the 10 headings point out of the arena; most policies leave through the boundary on them (so 8/10 is the usual ceiling), but a policy that turns at once can recover (TD3-dyn seed6 ep200: 10/10).

**Collapse diagnostic.** Best checkpoint run once on the static Corridor; fraction of steps whose heading change is exactly at the ±9° action limit, plus the mean training reward of the first/last 50 episodes.

| run | device | reward first50 → last50 | best ckpt (det.) | rand (best) | steps at ±9° cap | state |
|---|---|---|---|---|---|---|
| ARC seed0 | cpu | 18 → 69 | collision | – | 16% | learned |
| ARC seed1 | cpu | −19 → −15 | timeout | – | 100% (+9° every step) | **collapsed** (circles at start) |
| ARC seed2 | cpu | 48 → 81 | collision | – | 15% | learned |
| ARC seed3 | cpu | −22 → 86 | collision | – | 72% | learned |
| ARC seed4 | cpu | 23 → 117 | success | – | 52% | learned |
| TD3-dyn seed0 | cpu | 26 → 306 | collision | 5/10 | 22% | learned |
| TD3-dyn seed1 | cpu | 53 → 295 | success | 8/10 | 18% | learned |
| TD3-dyn seed2 | cpu | −59 → −61 | timeout | 0/10 | 100% | **collapsed** |
| TD3-dyn seed3 | cpu | −58 → −62 | timeout | 0/10 | 100% | **collapsed** |
| TD3-dyn seed4 | cpu | −56 → −24 | collision | 0/10 | 2.5% | partial |
| TD3 seed0 | cpu | −10 → −9 | timeout | 0/10 | 100% | **collapsed** |
| TD3 seed1 | cpu | −12 → −11 | timeout | 0/10 | 100% | **collapsed** |
| TD3 seed2 (CPU retrain; replaces the CUDA run) | cpu | −9 → −12 | timeout | 0/10 | 100% (+9°) | **collapsed** |
| TD3 seed3 | cpu | −10 → −11 | timeout | 0/10 | 100% | **collapsed** |
| TD3 seed4 | cpu | −13 → −13 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3 seed5 | cpu | −12 → −11 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3 seed6 | cpu | −14 → −11 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3 seed7 | cpu | −13 → −8 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3 seed8 | cpu | −13 → −11 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3 seed9 | cpu | −12 → −8 | timeout | 0/10 | 100% (+9°) | **collapsed** |
| TD3-dyn seed5 | cpu | −60 → −55 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3-dyn seed6 | cpu | 5 → 281 | collision | 0/10 (ep200: 10/10) | 12% | learned |
| TD3-dyn seed7 | cpu | −62 → −57 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3-dyn seed8 | cpu | −58 → −58 | timeout | 0/10 | 100% (−9°) | **collapsed** |
| TD3-dyn seed9 | cpu | 161 → 289 | collision | 0/10 (ep300: 8/10) | 7% | learned |

Measured:
- Collapse = the actor outputs the action bound at every step from the first episodes on; the agent circles at the start until the step limit (no collision, reward ≈ −10 static / ≈ −60 dynamic). All checkpoints of a collapsed run are identical in outcome.
- Collapsed seeds: ARC 1/5; **TD3-dyn 5/10** (seeds 2, 3, 5, 7, 8; seed 4 partial; learned 0, 1, 6, 9); TD3 static **10/10** (all CPU-trained). Static seeds 5–7, 8–9 and the seed-2 CPU retrain are in `out/checkpoints_corridor_td357.json`, `out/checkpoints_corridor_td389.json`, `out/checkpoints_corridor_td3s2cpu.json`; seed 2 was removed from `out/checkpoints_corridor_td304.json`.
- `out/checkpoints_corridor_td3dyn59.json` holds seeds 5–9.
- In the seeds that learned, the "best" checkpoint (highest training reward) is often not the best-performing one: TD3-dyn seed6 best = 0/10 but ep200 = 10/10; seed9 best = 0/10 but ep300 = 8/10. This is the same checkpoint oscillation seen for ARC (§2.15).
- Code check: `baselines.py` (last modified Mon 02:07) and `arc_2d/code` (Thu) predate every TD3 run; all TD3 runs used identical training code.
- Device: static seed 2 was first trained on CUDA and learned (reward 176 → 315, best ckpt success, 8/10). To keep every reported run on one device it was retrained on the CPU with the same seed and code; the CPU run collapsed like the other nine. All reported baselines are CPU-trained. The CUDA run is archived (`out/arc_models_gpu/td3_seed2`, `out/gpu_run_td3_seed2_checkpoints.json`) and is not in any table. CUDA and CPU runs with the same seed are not numerically identical, so the CUDA success is one draw, not evidence of a device effect.
- SAC-dyn (same hyperparameters, encoder, interface and training environment as TD3-dyn): 7/7 finished seeds learned (training reward by episode 100: 188–275), none collapsed; seeds 7–9 pending.
- TD3 differs from ARC's DDPG in both algorithm and hyperparameters (γ 0.99 vs 0.9, Adam 3e-4 vs RMSprop, batch 64 vs 32, buffer 50k vs 4000, noise 0.1). The higher collapse rate therefore cannot yet be attributed to the algorithm alone.
- GPU-trained checkpoints load and evaluate on the CPU (`map_location`); TD3 seed2 best reaches the goal in 232 steps.

### 4.6 TD3 on the Table 3/4 protocol (Corridor Gauntlet, 18 m/s, paired 50-environment dynamic suite) [FINAL]
`evaluate.py` → `out/final_corridor_td3dyn_a.json`, `…_td3dyn_b.json`, `…_td3_a.json`, `…_td3_b.json` (41 distinct checkpoints: best, final = ep300, oracle per seed). APF in all four files is 50/50 on the dynamic suite with outcomes identical to `final_corridor_a.json` (the ARC run), i.e. the APF baseline and the environment seeds are unchanged.

**Dynamic successes out of 50 per seed**

| variant | TD3 static (10 seeds) | TD3-dyn (10 seeds) | TD3-dyn pooled |
|---|---|---|---|
| best | 0 for every seed | 44, 42, 0, 0, 0, 0, 42, 0, 0, 29 | 157/500 |
| final (ep300) | 0 for every seed | 45, 40, 0, 0, 0, 0, 46, 0, 0, 48 | 179/500 |
| oracle | 0 for every seed | 45, 40, 0, 0, 0, 0, **50**, 0, 0, 48 | 183/500 |

- Collapsed seeds (all TD3 static; TD3-dyn 2, 3, 5, 7, 8) fail by collision in 48–49 of 50 environments (circling at the start until a moving obstacle hits them) and time out in the rest; TD3-dyn seed 4 (partial) fails by collision 39 / out of bounds 11.
- Static Corridor run (Table 3 protocol, fixed start, deterministic): TD3 static every checkpoint timeout; TD3-dyn final: success for seeds 0, 1, 6, 9; best: success for seed 1 only.

**Paired efficiency, TD3-dyn seeds that learned (environments both planners solve)**

| seed / ckpt | shared envs | APF eff. (mean ± SD) | TD3 eff. | APF L (m) | TD3 L (m) | TD3 more efficient in | envs only APF solves |
|---|---|---|---|---|---|---|---|
| s0 final | 45 | 71.8 ± 20.4 | 85.0 ± 7.3 | 1131 ± 1309 | 721 ± 67 | 37/45 | 5 |
| s1 final | 40 | 72.5 ± 18.8 | 92.8 ± 7.9 | 986 ± 713 | 661 ± 63 | 38/40 | 10 |
| s6 final | 46 | 74.7 ± 18.4 | 81.1 ± 4.1 | 1070 ± 1292 | 752 ± 38 | 36/46 | 4 |
| s6 oracle (ep200) | 50 | 72.7 ± 19.6 | 80.3 ± 4.4 | 1093 ± 1246 | 760 ± 58 | 33/50 | 0 |
| s9 final | 48 | 72.5 ± 20.0 | 86.5 ± 4.1 | 1106 ± 1271 | 704 ± 32 | 40/48 | 2 |

- No environment is solved by TD3 but not by APF (APF solves all 50).
- APF's path length on the shared environments is heavy-tailed (SD > mean in most rows): a few very long detours; TD3-dyn paths are tightly clustered (~660–760 m).
- For comparison, the corridor-trained ARC seeds (static training) reach 26, 0, 1, 26, 5 / 50 on the same suite (Table 4, §3.2).

### 4.7 SAC-dyn on the Table 3/4 protocol, seeds 0–6 [FINAL for 0–6; all 10 seeds in §4.8]
`eval_checkpoints.py` → `out/checkpoints_corridor_sacdyn_a.json` (seeds 0–3), `…_sacdyn_b.json` (4–6); `evaluate.py` → `out/final_corridor_sacdyn_a.json`, `…_sacdyn_b.json`. APF outcomes identical to `final_corridor_a.json` in both files (50/50).

**Oracle checkpoints (static Corridor, 10 random headings):** s0 ep300 10/10, s1 best 8/10, s2 ep150 6/10, s3 ep150 8/10, s4 ep200 8/10, s5 ep200 9/10, s6 ep250 8/10; deterministic run succeeds for every oracle.

**Dynamic successes out of 50 per seed (APF: 50/50)**

| variant | s0 | s1 | s2 | s3 | s4 | s5 | s6 | pooled |
|---|---|---|---|---|---|---|---|---|
| best | 40 | 48 | 44 | 49 | 41 | 38 | 39 | 299/350 (85%) |
| final (ep300) | 45 | 40 | 22 | 39 | 40 | 43 | 32 | 261/350 (75%) |
| oracle | 45 | 48 | 43 | 49 | 41 | 36 | 44 | 306/350 (87%) |

- No SAC seed collapsed; the worst variant of any seed still reaches 22/50 (s2 final).
- Failures are collisions (1–28 per model) and some out-of-bounds (0–12); no timeouts.
- Static Corridor (deterministic, Table 3 protocol): best succeeds for s1, s3, s4, s5, s6; final for s0, s4, s6.

**Paired efficiency, best checkpoint (environments both planners solve)**

| seed | shared | APF eff. | SAC eff. | APF L (m) | SAC L (m) | SAC more efficient in | only APF solves |
|---|---|---|---|---|---|---|---|
| s0 | 40 | 72.1 ± 20.3 | 92.6 ± 7.2 | 1148 | 661 ± 60 | 37/40 | 10 |
| s1 | 48 | 72.1 ± 19.7 | 83.4 ± 1.8 | 1110 | 730 ± 16 | 39/48 | 2 |
| s2 | 44 | 72.7 ± 20.6 | 89.4 ± 8.3 | 1127 | 687 ± 67 | 42/44 | 6 |
| s3 | 49 | 72.7 ± 19.8 | 86.7 ± 3.6 | 1099 | 702 ± 27 | 41/49 | 1 |
| s4 | 41 | 73.8 ± 18.5 | 89.0 ± 9.8 | 966 | 692 ± 84 | 32/41 | 9 |
| s5 | 38 | 75.2 ± 18.5 | 84.8 ± 6.6 | 1099 | 721 ± 53 | 35/38 | 12 |
| s6 | 39 | 73.0 ± 21.0 | 90.1 ± 8.8 | 1147 | 682 ± 75 | 32/39 | 11 |

- As for TD3-dyn: no environment is solved by SAC but not by APF; SAC is more efficient in 78–95% of the shared environments; SAC path lengths cluster at 660–730 m, APF's are heavy-tailed.

### 4.8 All learned planners: final numbers (10 SAC seeds; generated by `report.py` → `out/report.md`, "Learned baselines" and "Envelope summary") [FINAL]
**Training collapse** (`collapse_diagnostic.py` → `out/collapse.json`; collapsed = retained checkpoint at the ±9° bound on ≥ 99% of steps). Collapsed runs are at 100.0%; every other run is at ≤ 71.6% (ARC seed 3), so the threshold separates cleanly.

| learner (training) | seeds | collapsed | device |
|---|---|---|---|
| ARC (static) | 5 | 1 (seed 1) | CPU |
| TD3 (static) | 10 | 10 | CPU |
| TD3 (moving obstacles) | 10 | 5 (2, 3, 5, 7, 8); seed 4 not collapsed but 0/50 | CPU |
| SAC (moving obstacles) | 10 | 0 (fraction at bound 0.0–3.9%) | CPU |

Fisher exact, collapse TD3-dyn vs SAC: p = 0.033. Wilson 95% interval of the collapse rate: TD3-dyn 24–76%, SAC 0–28%.

**Table 3/4 protocol, corridor** (arc_best / oracle; dynamic successes /50, mean ± SD over seeds):

| learner | static arc_best | static oracle | dynamic arc_best | dynamic oracle | pooled η shared (learner vs APF) | learner more efficient |
|---|---|---|---|---|---|---|
| ARC (static), 5 | 1/5 | 4/5 | 11.6 ± 13.3 (0–26) | 18.6 ± 11.0 | 93.0 vs 71.3 | 53/58 |
| TD3 (static), 10 | 0/10 | 0/10 | 0.0 | 0.0 | – | – |
| TD3-dyn, 10 | 1/10 | 4/10 | 15.7 ± 20.7 (0–44) | 18.3 ± 23.8 | 86.0 vs 72.6 | 135/157 |
| SAC-dyn, 10 | 8/10 | 10/10 | 41.3 ± 6.9 (25–49) | 44.1 ± 4.0 | 88.0 vs 73.3 | 356/413 |

SAC per seed (arc_best): 40, 48, 44, 49, 41, 38, 39, 47, 42, 25; oracle: 45, 48, 43, 49, 41, 36, 44, 44, 42, 49.
- Seed-level exact permutation tests (arc_best): SAC vs TD3-dyn p = 0.0036; TD3-dyn vs TD3 static p = 0.087 (oracle: 0.0083 / 0.087).
- Per policy vs APF: no learned policy solves any corridor environment APF fails. 7/10 SAC seeds are significantly less reliable (exact McNemar p < 0.05); s1, s3, s7 fail only 2, 1, 3 APF-solved environments (p = 0.5, 1, 0.25). Per-seed Wilcoxon on shared efficiency: p ≤ 0.0042 for every SAC seed; ≤ 1.7e-5 for every learned TD3-dyn seed.
- Seed-mean shared path length: SAC 696 m vs APF 1061 m; TD3-dyn 715 vs 1120 m.
- **Pooled Wilcoxon p-values (report.md) pool pairs that share APF runs; the paper reports per-policy tests instead** (for ARC: p = 2.6e-6 and 2.1e-7 for s0 and s3 with 26 pairs each; s4 p = 0.0625 with 5 pairs, the minimum attainable).

**Envelope summary — % success, mean ± SD over ALL seeds of each learner** (a seed's 50-env/100-layout result where it exists, else its 10-env spot check):

| test | APF | APF, ARC constraints | ARC (5) | TD3 static (10) | TD3-dyn (10) | SAC-dyn (10) |
|---|---|---|---|---|---|---|
| noise 0 | 100 | 72 | 23.2 ± 26.6 | 0 | 31.4 ± 41.3 | 82.6 ± 13.8 |
| noise 1 m, 2° | 96 | 64 | 22.8 ± 24.8 | 0 | 31.0 ± 41.1 | 83.8 ± 12.5 |
| noise 2.5 m, 5° | 96 | 74 | 23.2 ± 20.6 | 0 | 30.4 ± 41.4 | 81.6 ± 13.4 |
| noise 5 m, 10° | 94 | 58 | 16.0 ± 14.6 | 0 | 23.2 ± 31.4 | 60.8 ± 17.6 |
| random layouts static | 68 | 90 | 38.0 ± 34.4 | 0 | 36.0 ± 46.8 | 94.8 ± 8.3 |
| random layouts dynamic | 98 | 80 | 25.4 ± 24.4 | 0 | 30.0 ± 39.0 | 75.6 ± 8.3 |
| speed 3 m/s (r 1.58) | 60 | – | 6.4 | 0 | 5.2 | 11.6 |
| speed 4.75 m/s (r 1.00) | 94 | – | 16.0 | 0 | 14.0 | 34.2 |
| speed 9 m/s (r 0.53) | 96 | – | 22.0 | 0 | 22.4 | 48.8 |
| speed 12 m/s (r 0.40) | 92 | – | 27.2 | 0 | 27.4 | 72.4 |
| speed 18 m/s (r 0.26) | 100 | – | 23.2 | 0 | 31.4 | 82.6 |
| speed 27 m/s (r 0.18) | 98 | – | 23.2 | 0 | 30.6 | 81.6 |

Consistency checks passed: noise level 0 and the 18 m/s sweep reproduce the Table 4 arc_best counts of every learned model exactly; APF sweep counts identical across all sweep files; APF outcomes identical in every final_corridor_* file.

**Spot checks for the 16 non-learning TD3 seeds** (10 TD3 static + TD3-dyn 2, 3, 4, 5, 7, 8; arc_best): 0 successes in `noise_collapsed10.json` (4 levels × 10 envs), `layouts_collapsed10.json` (10 static + 10 dynamic), `speed_sweep_corridor_collapsed10.json` (8 speeds × 10 envs) and `spot_urban_collapsed10.json` (6 TD3-dyn seeds, second layout, 10 envs).

**Random-layout successes where APF fails were re-run and verified** (e.g. layouts 2, 8, 15 for TD3-dyn s0: end distance 8.7–9.5 m < 10 m goal radius, min clearance 8.0–13.6 m > 5 m; APF times out on each).

### 4.9 Transfer of the moving-obstacle policies to the second layout [FINAL]
`out/final_urban_learned_a.json`, `…_b.json` (best + oracle of TD3-dyn 0, 1, 6, 9 and SAC 0–9); APF 48/50 on this suite (fails 2).

| learner | static (arc_best / oracle) | dynamic arc_best /50 | dynamic oracle /50 |
|---|---|---|---|
| ARC corridor → second (Table transfer) | 2/5 / 1/5 | 10.2 ± 10.4 | 10.8 ± 9.7 |
| TD3-dyn, 4 learned seeds | 4/4 / 4/4 | 37.0 ± 4.3 | 40.5 ± 5.1 |
| SAC-dyn, 10 seeds | 8/10 / 9/10 | 37.4 ± 10.2 (seed 9: 9/50) | 40.0 ± 2.5 |

The 10 SAC policies solve 15 (arc_best) / 17 (oracle) of the 20 (policy, environment) pairs formed by the 2 environments APF fails. "Final" rows in report.md for the urban layout cover only seeds whose oracle is ep300 and are not used.

### 4.10 Timing of the baselines [FINAL]
`timing.py --no-3d --out timing_baselines` (idle machine after all runs): per-step cost TD3-dyn 1.27–1.35 ms, SAC 1.31–1.40 ms, ARC (seed 4, same session) 1.28 ms, APF 0.39 ms. Compute per 100 m: learned 42–47 ms vs APF 21 ms (factor ~2, as in the paper's original idle timing 50.2 vs 24.2 ms). Training wall clock per run (7–10 runs in parallel on a thermally limited CPU, all cores ~1.6 GHz): TD3 8.8–16.3 h, TD3-dyn 5.5–13.9 h, SAC 7.8–17.2 h — load-dependent, not an algorithm property.

### 4.11 Corrections to earlier notes
- "2 of the 10 random headings leave the arena for every planner, so 8/10 is the maximum" was wrong: TD3-dyn seed 6 ep200 reaches 10/10. It is the usual, not a hard, ceiling.
- §4.4 collision wording corrected (see there).
- The one GPU-trained TD3 run (static seed 2) is archived in `out/arc_models_gpu/` and `out/gpu_run_td3_seed2_checkpoints.json`; the reported seed 2 is a CPU retrain (collapsed).

### 4.12 APF free-space traps, reproducible (`apf_traps.py` → `out/apf_traps.json`) [FINAL]
The paper's APF fails on 32/100 random static layouts, all timeouts, all solvable (another planner reaches the goal on each). Stall points: 178–555 m from the goal (median 438 m), 16.9–39.3 m from the nearest obstacle; resultant force ≤ 36% of the attractive force at every stall point; two-point oscillation in 27 of the 32; of the other 5, four show 6 distinct positions in the last 6 steps (a longer cycle or slow drift) and one (layout 99) is stationary. Layout 2: |F_att| = 2168, |F_rep| = 2209, |F_total| = 47. (Earlier note said "180–555 m" and implied a two-point oscillation in every case; corrected here.)

### 4.13 Independent audit of the paper's numbers (after the second revision round)
An independent pass checked ~290 numeric claims in `paper.tex` against the JSON outputs. Corrected in the paper: the 36/50 matched-APF figure needs all three constraints (rays + turn cap at 10 Hz gives 35); Narrow Passage efficiency for θ ≥ 50° is 60.8–66.9% (not 66.6–66.9%); the three gain pairs agree to within 6 mm (1.4 mm holds only for the first two); SAC per-seed Wilcoxon p < 0.005 (max 0.0042); APF shared-pair path ~1070 m; "never succeeds where APF fails" scoped to the custom suite (on the second layout the learned policies solve some of APF's two failures); the 15.3 m kernel-distance claim scoped to the five static layouts; transfer wording (best ckpt worse in both transferred rows). The stall analysis was made reproducible (§4.12).

### 4.14 Classical velocity-space baseline: Dynamic Window Approach (DWA, DWA-CV) [FINAL; in the paper]
`dwa.py` (planner), `dwa_tune.py` → `out/dwa_tune.json`, `dwa_eval.py` → `out/dwa_eval.json`, `dwa_report.py` → `out/dwa_report.json`, `dwa_timing.py` → `out/timing_dwa.json`.
- **Planner.** Unicycle; v ≤ ground speed (it may slow down and stop, starts from rest); 90°/s turn rate, 9 m/s², 360°/s²; 7×21 velocity samples over the dynamic window; 3 s constant-(v, ω) rollouts; inadmissible if the rollout comes within APF's 6 m margin, leaves the arena or cannot brake before the nearest obstacle; objective w_h·heading + w_c·clearance (capped at 30 m) + w_v·velocity, each term normalised over the admissible set. Limits and horizon scale with the ground speed in the speed sweep. Same information as APF (exact geometry of every obstacle now); DWA-CV also translates each obstacle along the rollout with the velocity estimated from its last displacement. Evaluation = harness protocol with APF's time budget (6000 decisions).
- **Kernel check.** Vectorised distance kernels agree with `apf.get_distance` to 1e−13 m on 1,200 random points over three layouts.
- **Tuning.** 3×4×3 grid of (w_h, w_c, w_v) on 20 held-out motion seeds (1000–1019, not evaluation seeds) + the static custom environment; selection by successes then efficiency. Selected: DWA (0.5, 0.2, 0.5), DWA-CV (0.5, 0.2, 0.2); both 21/21.
- **Static custom environment.** DWA: success, 841.5 m (72.3%); DWA-CV: success, 717.4 m (84.8%) (in-sample: part of the tuning set).
- **Paired dynamic suite (50 envs, 18 m/s).** DWA 47/50 and DWA-CV 47/50 (all failures collisions, in environments APF solves; exact McNemar vs APF p = 0.25 each). On the 47 shared successes: DWA η 62.7 ± 14.4 vs APF 73.3 ± 19.3 (DWA more efficient in 10, Wilcoxon p = 1.5e−4); DWA-CV 88.0 ± 18.4 vs 74.3 ± 17.5 (37/47, p = 7.3e−4). Mean speed on successes 17.3 / 15.2 m/s; time to goal 60.7 vs APF 60.6 s (DWA) and 49.3 vs 52.5 s (DWA-CV).
- **Against SAC (arc_best, 10 seeds).** DWA-CV's 47 successes exceed 7 seeds, equal 1 and are below 2 (48, 49); per-seed McNemar significant only for seeds 5 (p = 0.022) and 9 (p = 3e−6). On the 389 (seed, environment) pairs both solve: η 88.9 vs 88.1%, DWA-CV more efficient in 259. (No pooled test: pooled pairs reuse each DWA-CV run ten times.)
- **Second layout.** DWA 46/50, DWA-CV 48/50 dynamic; both succeed on the static layout.
- **Speed sweep (successes /50 at v = 3, 4, 4.75, 6, 9, 12, 18, 27 m/s).** DWA 7, 18, 26, 34, 43, 43, 47, 48; DWA-CV 23, 38, 40, 44, 41, 46, 47, 47 (APF 30, 37, 47, 46, 48, 46, 50, 49).
- **Noise (L0–L3).** DWA 94, 82, 82, 54%; DWA-CV 94, 92, 90, 70% (APF 100, 96, 96, 94%).
- **100 random layouts.** Static: DWA 99, DWA-CV 98 (APF 68); each solves 31 of APF's 32 stall layouts, together all 32. Dynamic: DWA 90, DWA-CV 99 (APF 98).
- **Timing (idle session; APF re-timed in the same session at 0.82 ms/step, 337 ms/run).** DWA 2.33 ms/step, 1109 ± 38 ms/run; DWA-CV 2.61 ms/step, 1147 ± 61 ms/run: 2.8× / 3.2× APF per step, 2.9× / 3.5× per 100 m. APF's own per-step cost varies by up to 2× between sessions on this thermally limited laptop (0.39–0.82 ms), so planners are compared within a session only.

### 4.15 Correction: run-to-run repeatability of static APF
Re-checked directly (ten runs with perturbation seeds 0–9 on each of the five static layouts): outcome and step count never change, path lengths agree to < 1e−10 m, but trajectories differ pointwise by up to 2.8 cm (Corridor Gauntlet 2.2 cm, Wall Maze 2.8 cm, Narrow Passage 1.2 cm). The earlier wording "identical to within 1e−13 m" described path length, not the trajectory; the paper now says "same outcome and step count, differ by at most 3 cm".
