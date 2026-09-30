# Paper edits: replacement text, section by section

Companion to `FINDINGS.md` (which holds the evidence). Text here is written to be pasted into the LaTeX source.
Every number here comes from `experiments/out/*.json` (see `FINDINGS.md` §3 and `out/report.md`). The only `{{...}}` left is where the authors must supply information that is not in the repository (where the 3D ARC runs were made).
Section numbers follow the PDF `Apf_research.pdf`.

---

## Abstract

Replace
> "... and static and dynamic volumetric 3D workspaces, with its gains and rotation angle selected by parameter sweeps."

with
> "... and static and dynamic volumetric 3D workspaces, using a single gain pair and a rotation angle fixed at 25° on every layout."

Replace
> "On the static 2D environment we run a direct, like-for-like comparison against an Actor-Recurrent-Critic (ARC) agent, an LSTM-augmented DDPG variant, on the same laptop with the same metrics."

with
> "On the static 2D environment we run a direct, like-for-like comparison against an Actor-Recurrent-Critic (ARC) agent, an LSTM-augmented DDPG variant, trained on the same layout and evaluated on the same machine under an identical protocol (start, goal radius, collision criterion and path metric)."

Replace the ARC-result sentences with the text given under "Abstract (results sentences)" near the end of this file.

Replace
> "The code and evaluation logs supporting this study are available in the GitHub Repository whose link is given at Section 9."

with
> "The code and evaluation logs supporting this study are publicly available (see Data Availability)."

(No `\ref` in the abstract.)

## 1. Introduction, contribution 4

> "4. All experiments run on consumer hardware (AMD Ryzen 7 5800HS laptop CPU, RTX 3050 4 GB laptop GPU, 16 GB memory), closer to plausible deployment constraints than the workstation-class setups commonly reported."

Add "The qualitative 3D ARC development of Section 5 was carried out separately {{authors: state machine / Kaggle}}." if that is where it ran.

## 2.3 Comparative Studies (first two sentences)

> "Controlled comparisons between classical and learned planners are rarer than the area's confidence suggests; a recent survey of learning-based navigation notes that there has been little direct comparison between the two paradigms [NEW: Xiao et al. 2022]. What such comparisons rarely check is whether both methods are being asked to do the same job."

New bibliography entry:
> X. Xiao, B. Liu, G. Warnell, and P. Stone, "Motion planning and control for mobile robot navigation using machine learning: a survey," *Autonomous Robots*, vol. 46, no. 5, pp. 569–597, 2022.

Delete the RRT and APF-vs-value-based-RL claims, and drop [18], [19] if they are no longer cited.

## 3.3 Repulsive component (clamp sentence)

Replace
> "The shortest distance is clamped from below at 0.8 units to prevent the 1/ρ² kernel from producing numerical singularities at obstacle surfaces."

with
> "As a numerical guard, the shortest distance is clamped from below at 0.8 m. With the 6 m collision margin of Section 3.5 enforced, the clamp was never active in any reported run: the smallest distance ever passed to the kernel was 15.3 m."

## 3.4 Obstacle distance kernels (triangle sentence)

Replace
> "if q lies inside, the distance is zero and the gradient points toward the nearest edge midpoint."

with
> "the inside case (distance zero) does not arise in practice: any run whose clearance falls below 5 m is terminated as a collision before the agent can enter an obstacle."

## 3.5 Motion integration (escape / step-count sentences)

Replace from "If a proposed step lands inside an obstacle ..." to the end of the subsection with:
> "If a proposed step comes within the 6 m collision margin of an obstacle, the planner makes up to five escape attempts, each a step of 1.8 s perpendicular to the current heading with small Gaussian noise. If all five fail, the candidate position is projected outward along the nearest obstacle's normal to 3 m beyond the margin. For evaluation, a run is recorded as a collision if the agent's exact clearance to any obstacle falls below 5 m after any step. A run succeeds when the agent comes within 10 m of the goal, and the path ends at that point. L therefore equals s times the step count unless an escape or projection step occurs. None occurred in any static 2D or 3D run; in the 50-environment dynamic suite at 18 m/s a single escape and a single projection step occurred over all runs. They become frequent only when obstacles are faster than the agent (Section 7.2b)."

## 3.7 Composite score (add after the definition of S)

> "A single gain pair (k_a, k_r) = (5.0, 0.85) is used on every layout. A 15 × 15 sweep of S on the custom environment finds a marginally shorter path at k_r/k_a ≈ 0.26 (736.2 m against 741.6 m), but that ratio lowers efficiency by 9–11 points on Dense Forest and Wall Maze, so we retain the cross-layout pair."

Also add:

> "Because the motion (7) follows the normalised force direction and the bias b is negligible relative to ‖F‖, the trajectory depends on the gains only through the ratio k_r/k_a: (5.0, 0.85), (2.5, 0.425) and (1.0, 0.17) produce paths that agree to within 1.4 mm. The (k_a, k_r) grid is therefore effectively a one-dimensional sweep over k_r/k_a."

## 4.1 Observation space

Replace
> "three navigational scalars (relative bearing to target χ, heading error ψ, and normalised goal distance d0) and nine ray-cast range readings distributed at 40° intervals around the agent."

with
> "three navigational scalars (the UAV heading ψ and the azimuth χ from the UAV to the target, both in the world frame and scaled to [0, 1], and the goal distance d0 normalised by 500 m) and nine ray-cast range readings, normalised by the 50 m sensor range, spread over the forward half-plane at 22.5° spacing (−90° to +90° relative to the heading)."

## 4.2 Reward

Replace the paragraph with:
> "The reward combines four terms. The line-of-sight term r_LOS = λ_LOS(π/2 − |χ − ψ|), with λ_LOS = 1.0, is given at every step and rewards heading toward the target. The obstacle term σ_obs‖Σ_i (1/d_i − 1/R_s) e_i‖², with σ_obs = −0.1, sensor range R_s = 50 m and e_i the unit vector of sensor i's angular offset, penalises nearby returns. The action term α_act|Δψ|, with α_act = −0.01, penalises heading changes linearly. The terminal reward is +100 on reaching the goal radius (10 m) and −100 on collision (any range below 5 m) or on leaving the arena."

## 4.3 Training configuration (first sentence)

> "We train for 300 episodes. In each episode five exploring-start rollouts, drawn uniformly from [30, 100]² m with a random initial heading, are run sequentially with a single shared actor-critic and a single FIFO replay buffer of capacity 4000. Once the buffer holds 32 transitions, one gradient update is made per environment step."

## 6.1 Shared simulation environment

Replace
> "a ten-obstacle configuration of three circles, three axis-aligned rectangles, and three triangles arranged to force the agent through two narrow passages. This environment is identical in both codebases; it is named urban_evaluation in the ARC repository and Corridor Gauntlet in the APF repository"

with
> "a ten-obstacle configuration of four axis-aligned rectangles (two 60 × 150 m blocks and two 60 × 60 m squares), three triangles and three circles. Two of the circles, at (70, 430) and (430, 70), are designated outliers and remain stationary in the dynamic suite. The layout is defined identically in both codebases (corridor_gauntlet in the ARC code and Corridor Gauntlet in the APF code)."

Fig. 1 caption: replace "Obstacle primitives: three circles, three axis-aligned rectangles, three triangles." with "Obstacles: four axis-aligned rectangles (two of them squares), three triangles and three circles; the two corner circles are the stationary outliers of the dynamic suite."

## 6.2 Dynamic obstacle motion model

Replace the first two sentences and the collision description with:
> "Each non-outlier obstacle is assigned a constant speed drawn uniformly from a type-specific range: rectangles 1.23–1.75 m/s, squares 2.10–3.00 m/s, circles 2.54–3.63 m/s and triangles 3.33–4.75 m/s (3D: boxes 1.58–2.25 m/s, spheres 2.54–3.63 m/s). The direction is drawn uniformly on the unit circle (unit sphere in 3D). Both planners move at 18 m/s: the APF step of 1.8 m spans 0.1 s and the ARC step of 3.0 m spans 1/6 s. Every obstacle is therefore at least 3.8 times slower than the agent, and moves at most 0.48 m per APF step or 0.79 m per ARC step. Obstacles reflect off the workspace walls. When two obstacles' bounding boxes overlap, they are separated along the axis of minimum overlap, their velocity components along that axis are exchanged, and each speed is then restored to its pre-contact value. An obstacle whose centre comes within 50 m of the goal is pushed away from it, so the goal region cannot be permanently occupied. Outlier obstacles remain stationary. Obstacle motion is integrated on a 1/30 s grid shared by both planners, so APF and ARC face an identical motion sequence for each seed; environment s uses velocity seed s (s = 0, …, 49)."

## 6.3 Hardware and software

> "All experiments reported in Sections 7.1–7.7 ran on a single laptop: AMD Ryzen 7 5800HS (8 cores, 16 threads), NVIDIA RTX 3050 Laptop GPU (4 GB), 16 GB DDR4, Windows 11; Python 3.12, NumPy 2.0, PyTorch 2.5, Shapely 2.1. The 2D ARC agent was trained on the CPU (one thread per seed; for this network size the GPU was slower). A 15 × 15 = 225-combination gain sweep on the custom environment took 9–27 min of wall time depending on the grid, dominated by combinations that exhaust the 6000-step budget; ARC was trained for 300 episodes per seed (112–139 min each with 3–6 seeds sharing the CPU; a non-learning seed took 353 min)."

## 6.4 Metrics (append)

> "Both planners are evaluated from q0 = (35, 35), with ARC's initial heading pointing at the goal. They share a 10 m goal radius, a 5 m exact-clearance collision criterion and a 10.8 km travel budget (6000 APF steps, 3600 ARC steps). Path metrics are computed only over runs that reach the goal."

## 5.5 / 6.1 / 8.3 — 3D ARC provenance (one sentence, wherever the 3D ARC arena is introduced)

> "The 3D ARC experiments of Section 5 were run separately from the 2D study {{authors: machine / Kaggle}}; the code in the repository is the scaled 500 × 500 × 500 m version of the 20 × 20 × 15 m arena (isotropic factor 25; 23 solid obstacles in the final stage: 10 spheres, 8 boxes, 5 cylinders)."

## 7.1 Static 2D: Head-to-Head on the Custom Environment (replace the whole subsection)

> "Table 3 reports the head-to-head on the custom environment. APF is summarised over 10 runs that differ only in the initialisation perturbation b. The trajectories are identical to within 10⁻¹³ m, so its spread is zero. ARC is summarised over five independently trained policies (seeds 0–4), each evaluated with its deterministic policy from q0 = (35, 35) with the initial heading pointing at the goal. The primary ARC model of each seed is the checkpoint the training pipeline itself retains (arc_best, the highest rolling 20-episode training reward). We also report the final checkpoint (episode 300), and, as an optimistic upper bound, the checkpoint that performs best on the static layout.
>
> Only one of the five primary policies reaches the goal; three collide after 222–378 m of flight and one (seed 1) never learned to make progress and circles until the 10.8 km budget is exhausted. With the final checkpoint the count is again one of five, and with the best checkpoint per seed four of five. Where an ARC policy does reach the goal it is more efficient than APF, but only modestly: 693–732 m against 741.6 m (83–88% against 82.0%). The ARC path hugs the western side of the layout at 3.0 m per step; APF's detours under the obstacle band (Fig. 1).
>
> The per-step inference cost is 1.51 ms for ARC and 0.44 ms for APF. Because ARC advances 3.0 m per step against APF's 1.8 m, it needs 33.3 planning steps per 100 m where APF needs 55.6. The compute actually spent is therefore 50.2 ms per 100 m against 24.2 ms, a ratio of 2.1 rather than the 3.5 suggested by the per-step cost alone. APF needs no training. The ARC figures are the output of a 300-episode training run per seed (112–139 min each on the laptop of Section 6; the non-learning seed 353 min), followed by a checkpoint-selection step whose outcome, as Section 7.3 shows, is far from reliable."

## Table 3 (replace)

| Metric | APF | ARC arc_best (primary) | ARC ep300 | ARC best checkpoint (upper bound) |
|---|---|---|---|---|
| Success | 10/10 runs | 1/5 policies | 1/5 | 4/5 |
| Path length (m) | 741.6 ± 0.0 | 699.0 | 696.0 | 710.3 ± 18.9 |
| Efficiency η (%) | 82.0 ± 0.0 | 87.0 | 87.4 | 85.7 ± 2.3 |
| Step displacement (m) | 1.8 | 3.0 | 3.0 | 3.0 |
| Steps | 412.0 ± 0.0 | 233 | 232 | 236.8 ± 6.3 |
| Steps per 100 m | 55.6 | 33.3 | 33.3 | 33.3 |
| Per-step cost (ms) | 0.44 | 1.51 | | |
| Compute per 100 m (ms) | 24.2 | 50.2 | | |
| Run time (ms) | 320 ± 13 | 423 ± 11 | | |
| Training required | none | 300 episodes per seed | | |

Caption: "Static 2D head-to-head on the custom environment (d_straight = 608.1 m). APF: mean ± SD over 10 runs. ARC: five independently trained policies, each evaluated deterministically from q0 = (35, 35) with the heading toward the goal; path metrics are over the policies that reach the goal. 'Best checkpoint' is selected on this layout and is therefore an optimistic upper bound. Per-step cost: APF force evaluation and margin check; ARC ray casting and network forward pass; both measured on an otherwise idle machine."

## 7.2 Generalisation: APF vs ARC on Shared Dynamic Environments (replace from "Panel A separates ..." to the end of the subsection)

> "Both planners fly at the same ground speed, 18 m/s. This is APF's 1.8 m per 0.1 s step, and ARC's 3.0 m step is advanced 1/6 s of obstacle motion. The obstacles (1.2–4.75 m/s) are therefore always slower than the agent (Section 6.2).
>
> Panel A separates the two planners sharply. APF succeeds in all 50 environments (95% CI 93–100%) with no collision; the closest it comes to an obstacle is 5.66 m, and it needs one escape step in 50 runs. The five static-trained ARC policies succeed in 26, 0, 1, 26 and 5 of the 50 environments (11.6 ± 13.3; 58 of 250 pooled runs), and collide in 185 of the 250 runs. No ARC policy, in any checkpoint variant, succeeds in an environment that APF fails. The ARC successes are a strict subset of APF's, and an exact McNemar test rejects equal success rates for every policy (p ≤ 1.2 × 10⁻⁷).
>
> Panel B shows that where ARC does reach the goal, it keeps the efficiency advantage seen in the static case, and here it is large. On the 58 (policy, environment) pairs both planners solve, ARC's path averages 656 m against APF's 1155 m, a mean efficiency of 93.0 ± 5.6% against 71.3 ± 21.7%. ARC is the more efficient planner in 53 of the 58 pairs (Wilcoxon signed-rank p = 2 × 10⁻¹⁰). APF's large spread comes from a minority of environments in which moving obstacles repeatedly cut across its route and force long detours (efficiency 7–99% over the 50 environments). The reading is again the operating-envelope picture: the analytical planner trades path quality for near-total robustness to motion it was never tuned for, while the learned policy is efficient on the instances it solves but has no robustness to the perturbation, and whether it solves them at all depends mainly on which training seed produced it."

## Table 4 (replace)

Panel A — outcomes, all 50 environments

| | APF | ARC (5 policies) |
|---|---|---|
| Success | 50/50 (100%) | 11.6 ± 13.3 per policy (26, 0, 1, 26, 5); 58/250 pooled |
| Collision | 0/50 | 185/250 |
| Step-limit timeout | 0/50 | 2/250 |
| Left the arena | 0/50 | 5/250 |
| Solved by this planner only | 192 pairs | 0 |

Panel B — paired, 58 shared-success (policy, environment) pairs

| | APF | ARC |
|---|---|---|
| Path length (mean ± SD, m) | 1155.0 ± 1275.3 | 656.3 ± 43.5 |
| Efficiency η (mean ± SD, %) | 71.3 ± 21.7 | 93.0 ± 5.6 |
| More efficient in | 5/58 | 53/58 |

Caption: "Paired APF-vs-ARC comparison over 50 shared dynamic environments. Both planners move at 18 m/s through the identical seeded obstacle motion, with a 10 m goal radius, a 5 m exact-clearance collision criterion and a 10.8 km travel budget. ARC: the five corridor-trained arc_best policies. Panel B pools the (policy, environment) pairs that both planners solve; no failed or truncated trajectory enters a path average. ARC's successes are a strict subset of APF's (McNemar p ≤ 1.2 × 10⁻⁷ per policy); ARC's efficiency advantage on shared successes: Wilcoxon p = 2 × 10⁻¹⁰."

## 7.2b New subsection — The speed-ratio envelope (insert after 7.2, with Fig. `fig_speed_envelope.png`)

> "The comparison above holds the agents at 18 m/s, where every obstacle is at least 3.8 times slower than the agent. To locate the edge of each envelope we repeat the paired suite at eight common ground speeds from 3 to 27 m/s. That sweeps the ratio r of the fastest obstacle speed to the agent speed from 1.58 down to 0.18, over the same 50 seeded environments (Fig. X). APF succeeds in 92–100% of environments for every r ≤ 1. It degrades once obstacles become faster than the agent: 74% at r = 1.19 and 60% at r = 1.58. ARC stays below APF at every ratio (mean 3.2–13.6 successes out of 50). It declines toward r > 1 too, but the spread between training seeds (0 to 36 out of 50 at a single ratio) is larger than the effect of speed. The robustness of the analytical planner to obstacle motion is therefore itself bounded, and the bound is the relative speed of agent and obstacles. An evaluation that lets the classical planner move faster than the learned one, relative to the same obstacles, would overstate the gap between them."

Figure caption: "Operating envelope in the dynamic suite: success over 50 paired environments versus the ratio of the fastest obstacle speed to the agent speed (both planners at the same speed). Thin lines: individual ARC training seeds; band: ±1 SD; dotted line: r = 1."

## 7.2c New subsection (short) — Transfer across layouts (insert after 7.2b)

> "To separate in-distribution performance from transfer, we also trained five ARC policies on a second layout (the layout of the original ARC implementation) and evaluated all ten policies on both layouts (Table X). On the static tests the expected in-distribution effect appears only for the best checkpoints: 4/5 and 5/5 in-distribution against 1/5 and 0/5 transferred. In the dynamic suites the seed-to-seed spread (SD of 7–13 environments out of 50) is as large as any layout effect. APF, which needs no training on either layout, succeeds in 10/10 static runs and in 50/50 and 48/50 dynamic environments."

Table X (transfer):

| ARC trained on → evaluated on | static, arc_best | static, best checkpoint | dynamic, arc_best (/50) | dynamic, best checkpoint (/50) |
|---|---|---|---|---|
| custom → custom | 1/5 | 4/5 | 11.6 ± 13.3 | 18.6 ± 11.0 |
| custom → second layout | 2/5 | 1/5 | 10.2 ± 10.4 | 10.8 ± 9.7 |
| second layout → custom | 1/5 | 0/5 | 16.0 ± 10.0 | 31.2 ± 4.0 |
| second layout → second layout | 2/5 | 5/5 | 12.0 ± 6.9 | 26.2 ± 9.7 |
| APF (no training): custom / second | 10/10 / 10/10 | | 50/50 / 48/50 | |

## 7.3 ARC Training Dynamics (replace)

> "Fig. 5 shows the five 300-episode training runs on the custom environment. Rollout success rates are quantised in steps of 0.2 because each episode aggregates five exploring-start UAVs; the episode reward is the mean of their continuous returns and is not quantised. Four seeds learn to reach the goal intermittently: over the last 50 episodes their rollout success is 0.12–0.34 and their collision rate 0.54–0.74. Seed 1 never learns to make progress: its rollouts neither reach the goal nor collide, and they time out at the 500-step limit. The same pipeline on the second layout reaches rollout success of 0.59–0.66, so the custom environment is markedly harder for this agent.
>
> Rollout success is measured under exploration noise and random starts, so it is not the evaluation result. Evaluating every saved checkpoint deterministically shows a more important effect: policy quality oscillates during training. A typical seed goes from 0/10 to 8/10 successes from random initial headings between episodes 50 and 150, back to 0/10 at episodes 200–250, and partially recovers by episode 300. This pattern appears in every seed that learns, on both layouts. The checkpoint retained by the pipeline, chosen by training reward, is usually not the one that evaluates best. This is the instability of single-critic DDPG with fixed exploration noise and a small FIFO buffer, the same mechanism that Section 5 identifies in 3D, and it is already present in 2D. For a practitioner it adds a model-selection step, with its own evaluation cost, to the setup cost of the learned planner."

## Table 5 (replace)

| Seed | Rollout success, last 50 episodes | Rollout collision, last 50 | Episode reward, last 50 | Wall clock (min) |
|---|---|---|---|---|
| 0 | 0.12 | 0.73 | 68.5 | 122 |
| 1 | 0.00 | 0.00 | −14.7 | 353 |
| 2 | 0.18 | 0.74 | 80.5 | 122 |
| 3 | 0.26 | 0.63 | 85.7 | 139 |
| 4 | 0.34 | 0.54 | 117.0 | 120 |

Caption: "ARC training on the custom environment, 300 episodes per seed, 3–6 seeds sharing the CPU with one thread each. Seed 1 never learns; its 500-step rollouts make it the slowest run."

## 7.4 Rotation Angle Sensitivity (replace the second and third paragraphs)

> "On this layout the sweep does not favour large angles. Efficiency is highest for pure radial repulsion (84.9% at θ = 0°), dips to 80.6% at 15°, is 82.0% at the adopted 25°, stays at 82.4–83.6% from 30° to 80°, and falls to 74.2% at 90° (Fig. 6). Every angle reaches the goal on this layout. Across the other layouts the response is not smooth. Wall Maze fails at 0–10° and again at 55°; Dense Forest and Scattered Mixed show jumps of 7–11 points between neighbouring angles; and Narrow Passage fails for every θ ≤ 45° and succeeds for every θ ≥ 50°. We retain θ = 25° as a value fixed before the sweep, not tuned on it. The sweep shows that no single angle is best on every layout, and it also shows that the Narrow Passage failure reported in Section 7.5 is a property of θ = 25°, not an inherent limit of fixed-handedness rotation: every θ ∈ {50°, 60°, …, 90°} clears all five layouts."

Fig. 6 caption: "Trajectories on the custom environment for θ = 0°–90° in 15° steps, with path efficiency. The response is non-monotonic; pure radial repulsion (θ = 0°) gives the shortest path on this layout."

## 7.5 APF Robustness Across Layouts (edit)

- "Wall Maze (8 rectangular barriers)" → "Wall Maze (8 obstacles: five wall segments, two squares and a triangle)".
- "Four of the five layouts are cleared cleanly, with a mean efficiency of 82.3%" → "... 83.3%".
- Append: "At θ ≥ 50° the planner clears Narrow Passage as well (Section 7.4)."

## Table 6 (replace)

| Layout | Success | Steps | L (m) | η (%) |
|---|---|---|---|---|
| Custom environment | 10/10 | 412 | 741.6 | 82.0 |
| Dense Forest | 10/10 | 393 | 707.4 | 86.0 |
| Wall Maze | 10/10 | 387 | 696.6 | 87.3 |
| Scattered Mixed | 10/10 | 434 | 781.2 | 77.8 |
| Narrow Passage | 0/10 | 6000 | 10800 | 5.6 |

Caption: "APF on five static 2D layouts (APF only), 10 runs each (identical to within 10⁻¹³ m). One gain pair, (k_a, k_r) = (5.0, 0.85), and θ = 25° on every layout. All rows share s = 1.8 m and d_straight = 608.1 m, and L = s × steps exactly (no escape step occurred)." Delete the old footnote about escape deflections.

## 7.6 Three-Dimensional Results (replace)

> "The 3D APF planner was evaluated in static and dynamic configurations of the volumetric counterpart of the custom environment, 10 runs each (Table 7). It reaches the goal in every run, with an efficiency of 91.3% (static) and 90.9 ± 3.4% (dynamic). That is higher than the planner's 2D efficiency on the custom environment (82.0% static, 72.7 ± 19.6% dynamic), despite the longer straight-line distance. The most plausible cause is the extra dimension, which lets the path pass over or under obstacles that it must go around in 2D, rather than any property of the 3D extension.
>
> To test the formulation on a 3D world it was not designed for, we also ran it, unchanged, inside the arena used for the 3D ARC agent (Section 5). The only addition was a cylinder distance kernel (Table 7b). Under that arena's own success (20 m) and collision (1.75 m) criteria, and with and without its stochastic wind disturbance, APF succeeds in all 60 runs across the three curriculum stages. That includes the final stage with 23 solid obstacles, where it achieves 77.1% efficiency and keeps at least 27 m clearance. We do not report a 3D APF-vs-ARC head-to-head: the trained 3D ARC policies are not part of this study's artefacts, and the large-arena 3D ARC training is seed-unstable (Section 8)."

## Table 7 (replace)

| Metric | 3D static | 3D dynamic |
|---|---|---|
| Success | 10/10 | 10/10 |
| Steps (mean) | 453 | 455.6 ± 18.4 |
| Path length (m) | 815.4 | 820.1 ± 33.1 |
| Efficiency η (%) | 91.3 | 90.9 ± 3.4 |
| Wall time (mean, ms) | 232 ± 33 | 397 ± 15 |

## Table 7b (new) — APF in the 3D ARC arena

| Stage | Solid obstacles | Wind | Success | Path length (m) | η (%) | Min clearance (m) | Wall time (ms) |
|---|---|---|---|---|---|---|---|
| 0 | 4 | off | 10/10 | 703.8 | 96.2 | 31.6 | 101 |
| 0 | 4 | on | 10/10 | 704.9 ± 1.1 | 96.1 ± 0.1 | 31.3 | 105 |
| 1 | 15 | off | 10/10 | 788.4 | 85.9 | 19.8 | 520 |
| 1 | 15 | on | 10/10 | 789.1 ± 2.4 | 85.8 ± 0.3 | 19.7 | 540 |
| 2 | 23 | off | 10/10 | 877.7 ± 0.9 | 77.1 ± 0.1 | 31.8 | 887 |
| 2 | 23 | on | 10/10 | 872.9 ± 26.1 | 77.6 ± 2.3 | 27.0 | 905 |

Caption: "APF (unchanged gains and θ) inside the 500³ m arena of the 3D ARC agent (start (25, 25, 25), goal (462.5, 462.5, 300), d_straight = 677.1 m), under that arena's success (≤ 20 m) and collision (SDF < 1.75 m) criteria; wind is the arena's own stochastic disturbance."

## 7.7 Computational Cost (replace)

> "On the Ryzen 7 5800HS laptop the APF planner completes a static 2D run in 320 ms (10-run mean), a 3D run in 232 ms (static) or 397 ms (dynamic), and a run in the 23-obstacle 3D ARC arena in 887 ms. The per-step planning costs are 0.44 ms for APF and 1.51 ms for ARC, a factor of 3.5. Normalising by path length as in (10) and (11), APF spends 24.2 ms of compute per 100 m of trajectory and ARC 50.2 ms, a factor of 2.1. The learned planner is therefore about twice as expensive per unit distance at inference time. The larger cost asymmetry is still the training phase APF does not have: 300 episodes, or about two hours per seed on this machine, plus the checkpoint selection of Section 7.3. The 2D ARC network is small enough (batch 32, 128-unit LSTM) that training on the laptop's GPU was slower than on one CPU thread (30.8 ms against 20.2 ms per update), so all 2D training used the CPU."

## 8.1 What the Results Suggest (replace the first paragraph)

> "The defensible reading is not that one method is better. It is that the two occupy different operating regions. On the static environment the learned policy, when it works, produces a path 1–6 points more efficient. But only one of five training runs produces a primary policy that works, and its inference compute per metre travelled is about twice APF's. In the dynamic suite the analytical planner holds 100% success at equal speed, while the static-trained policies succeed in 0–52% of the environments and never in one the analytical planner fails. Where they do succeed they are markedly more efficient (93% against 71%). The analytical planner's own envelope ends where obstacles become faster than the agent (Section 7.2b), and it extends unchanged to two independently designed 3D worlds. ARC is the more efficient planner on the instances it solves; APF is the portable and reliable one."

## 8.2 Why Rotational Deflection Does Not Solve Narrow Passage (replace the angle sentences)

> "... In Narrow Passage the symmetry is four-fold, and the tangential components from opposing obstacles partially cancel at θ = 25°. The rotation sweep shows that a larger fixed angle does break it: every θ ≥ 50° clears Narrow Passage (66.6–66.9% efficiency) and, apart from θ = 55°, where Wall Maze fails, also clears the other four layouts. The failure is therefore specific to our conservative choice of θ, not to fixed-handedness rotation as such. An adaptive scheme for θ conditioned on the local symmetry of the force field, as in [10, 11], would remove the need to choose."

## 8.3 Limitations (replace the first three paragraphs)

> "The head-to-head uses one environment. ARC was trained and evaluated on it (a second layout enters only through the transfer experiment), and five seeds per layout is enough to show the seed sensitivity but not to estimate it precisely.
>
> The learned baseline is a DDPG-based agent with the training budget and hyperparameters of the original ARC formulation. Its checkpoint-to-checkpoint instability means that better model selection, longer training or a TD3-style critic would probably raise its numbers. We report the pipeline's own selection as primary and the best checkpoint as an upper bound, rather than tuning the baseline.
>
> The paired dynamic comparison uses one obstacle motion model and a common ground speed. Section 7.2b shows that the conclusion depends on the ratio of obstacle to agent speed, so results at other ratios should be read from Fig. X, not from Table 4."

Keep the 3D ARC and hardware limitation paragraphs, updating "8 GB RAM, 4 GB VRAM" to this machine where it refers to the 2D study, and stating where the 3D ARC runs were made.

## 9. Conclusion (replace the first paragraph)

> "This paper characterised the operating envelope of a goal-scaled APF planner with rotational deflection across static, dynamic and volumetric navigation tasks. It included a like-for-like comparison against an LSTM-augmented DDPG agent on a shared static layout and a paired, equal-speed comparison on 50 dynamic environments. When the learned agent reaches the goal it is the more efficient planner (83–88% against 82% static, 93% against 71% on shared dynamic successes). It reaches the goal far less reliably: only one of five primary policies solves its own training layout, the policies succeed in 0–52% of the dynamic environments, and they never succeed where APF fails. The analytical planner holds 100% at equal speed, degrades only once obstacles outpace the agent, and runs unchanged in two independent 3D worlds, at about half the per-metre inference compute and with no training."

## Abstract (results sentences; replace "The learned agent produces shorter ... more efficient where it does succeed.")

> "On the shared static environment the learned agent's paths are 1–6 points more efficient when it reaches the goal, but only one of five independently trained policies does, and its inference compute per metre is about twice the analytical planner's. On 50 paired dynamic environments, with both planners at the same speed, the static-trained policies succeed in 23% of runs (0–52% per policy) against the analytical planner's 100%, never solving an environment the analytical planner fails, while remaining markedly more efficient where they succeed (93% against 71%). Sweeping the obstacle-to-agent speed ratio locates the analytical planner's own envelope edge where obstacles become faster than the agent. The analytical planner also runs unchanged in two independently designed 3D worlds."

## Figure captions (3, 4, 5)

- Fig. 3: "An environment both planners solve (environment 14, ARC seed 0). Filled shapes: obstacles at t = 0; dashed outlines: obstacles when ARC reaches the goal. ARC takes the direct route through the moving field; APF detours along the southern and eastern edges."
- Fig. 4: "The more common outcome (environment 7, ARC seed 0). The static-trained policy collides with a moving obstacle; APF reaches the goal. The five ARC policies collide in 185 of their 250 runs."
- Fig. 5: "ARC training on the custom environment, five seeds (grey) and their mean (red): rollout collision rate (top), episode reward (middle) and rollout success rate (bottom), all under exploration noise and random exploring starts. Rates are quantised in steps of 0.2 (five UAVs per episode); the reward is not. Seed 1 never learns (success and collision both zero)."


## 8.4 Future work: add an envelope-informed hybrid paragraph

> "The measured envelopes also suggest a concrete hybrid. The learned policy is the more efficient planner inside its training distribution and when obstacles are slow relative to the agent; the analytical planner degrades more gracefully when obstacle speeds approach the agent's and needs no retraining when the layout changes. A switching planner that executes the learned policy by default, and hands control to the APF field when the measured clearance falls below a threshold or the closing speed of the nearest obstacle exceeds a fraction of the agent's speed, would use the envelope boundaries of Section 7 directly as its switching rule. Evaluating such a scheme against existing APF–DRL hybrids (APF-shaped rewards, residual policies and safety-shield switching) on the paired suite is left to future work."

## Data Availability

> "The code, trained models and evaluation logs supporting this study are available at https://github.com/Adi-Narayan/Research-Paper. The experiments in Section 7 are reproduced by the scripts in the experiments/ directory."

---

# Second revision round (applied directly in `paper.tex`; evidence in `FINDINGS.md` §4)

New material:
- §4.4 *Additional Learned Baselines: TD3 and SAC* (new; Table `tab:baselines` hyperparameters; training distributions and seed ranges; definition of training collapse).
- §6.4 Metrics: seed counts per learner; per-policy McNemar/Wilcoxon; seed-level exact permutation and Fisher tests (new refs `henderson2018`, `agarwal2021`).
- §6.3 Hardware: baselines trained on CPU, ≤10 parallel runs, thermal limiting; one GPU-trained TD3 run retrained on CPU.
- §7 *Stronger Learners and Moving-Obstacle Training* (new; Table `tab:baselines-results`, Figs. `baselines_seeds.png`, `baselines_training.png`).
- §7 *APF Under the Learned Agent's Constraints* (new; Table `tab:matched`).
- §7 *Sensing Noise and Unseen Layouts* (new; Table `tab:envelope`; APF free-space traps and the turn-cap factorial).
- Speed-ratio, transfer, training-dynamics, APF-robustness and computational-cost sections extended with the baselines; Fig. `speed_envelope.png` redrawn with TD3/SAC.

Rewritten: abstract, highlights (commented), keywords (+SAC), introduction (asymmetry paragraph, study description, contributions 1–4), Discussion §8.1, Limitations (training distribution, seed counts with Wilson intervals, untuned baselines, spot checks), Future work (training across speed ratios; hybrid kept as future work), Conclusion.

Corrections to text that was already in the paper:
- Table 4 efficiency test: the pooled Wilcoxon p = 2e-10 pooled pairs that share APF runs; replaced by per-policy tests (p = 2.6e-6, 2.1e-7; 0.06 at n = 5).
- APF robustness: "four of five layouts" now qualified by the 32/100 random-layout traps.

Unchanged by request: Acknowledgements.
New references: `sac`, `sacapp`, `henderson2018`, `agarwal2021`.
