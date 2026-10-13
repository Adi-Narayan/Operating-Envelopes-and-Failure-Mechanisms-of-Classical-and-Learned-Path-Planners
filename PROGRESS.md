# PROGRESS

Plan: `instructions.md` (kept outside the repository, in the paper folder; not committed).
Paper source: `paper.tex` in the paper folder (outside the repository).

## Status
- **Section 3 (verification pass): complete, all checks passed.** Nothing blocks the workstreams.
- `[ASK ADI]` open: (1) the C2 math in `notes/c2_proposition.md`; (2) every new reference (WS-G) before it goes in.
- Job queue: `b45` (B4 TD3-fixed + B5 SAC-static training, 20 jobs, 7 at a time) and `d3` (arbiter threshold tuning, 90 jobs, 2 at a time) running.

## Log

### 2026-10-12: Section 3 verification pass
Code at commit `1e801cf` (HEAD before this block). Machine idle (no training) throughout; one thread per process.

**3.1 Reproduce before extending** (`scripts/verify_section3.py reproduce` -> `results/v2/verification/reproduce/`)
| Check | Paper | Reproduced | Run-by-run against stored results |
|---|---|---|---|
| APF, 50 dynamic environments, 18 m/s | 50/50, mean eta 72.7% | 50/50, 72.74% | 50/50 identical (path length difference 0.0) |
| SAC moving seed 0 (`sacdyn_seed0/arc_best`), dynamic suite | 40 (per-seed report) | 40 | 50/50 identical |
| APF, 100 random static layouts | 68/100, 32 timeouts | 68/100, 32 timeouts | 100/100 identical |
| DWA-CV, dynamic suite | 47/50 | 47/50 | 50/50 identical |

**3.2 Paper and code consistency**
- Section 4.3 "Only the learning algorithm changes" was wrong. Code (`experiments/baselines.py` `BASELINE_TRAIN_CONFIG`, `arc_2d/code/config.py`, `arc_2d/code/networks.py`) shows TD3/SAC differ from ARC in: optimiser and learning rate (Adam 3e-4 vs RMSprop 1e-3), discount (0.99 vs 0.9), soft-update rate (0.005 vs 0.01), batch (64 vs 32), replay capacity (50,000 vs 4,000), TD3 exploration noise (0.1 vs 0.2 of the action range; SAC samples), and encoder sharing (one encoder copy per actor and critic vs ARC's single shared LSTM). Gradient-norm clipping at 1.0 is **common to all three** (ARC clips too), so the paper's statement that clipping was "for CPU training" was also wrong. **Fixed in paper.tex** (Section 4.3, Table C.2 caption, Limitations).
- Collapse definition: `experiments/collapse_diagnostic.py` matches Section 4.3: retained checkpoint (`arc_best`), deterministic run on the static custom environment from q0 heading towards the goal, fraction of executed heading changes within 0.05 deg of the 9 deg bound, collapsed if >= 0.99.
- `arc_best` selection: the same rule for ARC, TD3 and SAC (`arc_2d/code/train.py`: highest mean training reward over the last 20 episodes, tracked from episode index 20, i.e. after 21 episodes; TD3/SAC reuse `ARCTrainer` with the agent swapped in `experiments/train_arc_seed.py`).

**3.3 Seed and layout map**: `notes/seed_map.md`.
- DWA tuning seeds found: motion 1000-1019 on the custom environment + the static custom environment (`experiments/dwa_tune.py`).
- New held-out sets defined, no collisions: motion 500000-500049 (APF perturbation 501000-501049, noise 502000-502049); tuning layouts `make_layout(500000-500049)` with motion 510000-510049 and dynamic APF perturbation 520000-520049. All 50 tuning layouts generate; none duplicates an evaluation layout.

**3.4 Scale-invariance audit**: `notes/scale_invariance_audit.md`; `scripts/verify_section3.py scaleinv`, `scripts/verify_scaleinv_detail.py` -> `results/v2/verification/scale_invariance/`.
- Every time-carrying quantity is tied to the decision period (obstacle tick = APF period / 3; goal push-away per tick; DWA limits scale with v; budgets and noise per decision).
- Bit-identity test (motion seeds 0-9; 18 m/s vs 9 m/s with obstacle speeds x0.5): SAC bit-identical 10/10; APF same outcome and step count 10/10, max pointwise difference 4.8 cm, traced to floating-point rounding in obstacle positions (<= 2.3e-8 m) amplified by the field, not to a time dependence. **Passed.**

### 2026-10-12 evening: Day 1 work blocks
**Infrastructure.** `scripts/jobqueue.py` (resumable queue: `.done` markers in `results/v2/queue/<name>/`, RAM gate, below-normal priority for evaluation), `scripts/v2common.py` (manifests). Named `jobqueue.py`, not `queue.py`, because a `queue.py` in `scripts/` shadows Python's `queue` module and breaks torch imports.
- Free RAM is the binding constraint (1.2-2.5 GB free with Adi's applications open; a training process uses ~0.36 GB, an APF job ~0.08 GB). Training runs 4 at a time, not 20.

**WS-B code changes (opt-in; defaults unchanged).** `baselines.py`: `Actor.pre()`, TD3 `pretanh_penalty` (default 0), logging of E[u^2] and |u| > 3 (TD3, SAC). `train_arc_seed.py`: `--gamma`, `--exploration-noise`, `--pretanh-penalty`, `--out-root`, per-episode `train_log.jsonl`. **Identity check passed:** with default flags, TD3 moving seed 0 reproduces the stored run's episode rewards 1-3 exactly (`results/v2/ws_b/checks/`).

**B3 pre-requisites.** 50-episode proxy valid: collapse at the ep-50 checkpoint matches the final status in 9/10 TD3-moving and 10/10 TD3-static seeds (`results/v2/ws_b/proxy_check/`). lambda = 2.0e-3 from a 10-episode TD3 run, seed 100 (`results/v2/ws_b/lambda_calibration/`); the pre-tanh output is already saturated at episode 0 (95% of |u| > 3). `prereg.md` B3/B4/B5 committed (`39a16ca`) before any B3 run.

**Running.** B3 (12 jobs, `jobs/b3.json`), B1 (`jobs/b1.json`), C3 theta sweep (`jobs/c3.json`), C4 tuning (`jobs/c4.json`; prereg `f762095`).

**C1 done** (`results/v2/ws_c/c1_classify/`): of the 32 stalls, 27 period-2 orbits around stable nodes, 4 non-periodic orbits around stable foci, 1 arena-boundary trap (layout 99: pinned at x = 500 m, force points out of the arena; the paper's "all in open space" needs this exception). curl/div = tan(theta) at every equilibrium to 1e-9; derived in `notes/c2_proposition.md` (Lemma 1: exact for goal exponent n = 2). **[ASK ADI]** check the C2 math before it goes into the paper.

**C4.** `experiments/apf_damped.py`; reproduces plain APF and the always-on cap path for path (`results/v2/ws_c/c4_identity/`).

**WS-D.** `experiments/arbiter.py` (L = learned, C = APF with ARC's rays at 6 Hz, or DWA-CV). Reproduces SAC alone, ray-APF alone and DWA-CV alone path for path on 10 environments (`results/v2/ws_d/identity/`).

### 2026-10-12 night: Day 1 outcomes
**B3 done; rule applied** (`scripts/ws_b_b3_select.py`, outcome appended to `prereg.md`, `c7235e6`). Collapsed ep-50 checkpoints, TD3 with moving obstacles, seeds 10-12 (3 per condition): (a) baseline 2, (b) gamma 0.9 2, (c) pre-tanh penalty 0, (d) exploration noise 0.3 3. Candidate (c) has strictly fewer collapses than (a): **B4 runs with `--pretanh-penalty 0.002`.** Under (c) E[u^2] over the last 10 episodes is 3.6-4.9 and 11-21% of pre-tanh outputs have |u| > 3; collapsed runs reach E[u^2] 146-4,847 with 100% |u| > 3.

**B2 done** (`results/v2/ws_b/b2_returns/`). Discounted return of fixed behaviours, gamma = 0.9 / 0.99. Static: circling -0.82 / -4.78, greedy 8.79 / -2.61, random -5.19 / -52.68, SAC 6.90 / 72.28. Moving: circling -0.82 / -10.28, greedy 8.39 / 0.70, random -5.18 / -54.47, SAC 6.88 / 85.40. Circling never has a higher return than heading to the goal.

**B1 done** (`results/v2/ws_b/b1_summary.json`, figure `results/v2/ws_b/figures/collapse_mechanism.pdf`). Collapsed TD3 (static): 100% of actions at the bound, 100% of pre-tanh outputs |u| > 3 from the first checkpoint; critic Q(bound) = -0.32 > Q(zero) = -1.12 in 70% of states, Monte Carlo return -0.60. Collapsed TD3 (moving): Q(bound) -5.02, Q(zero) -6.15, return -12.65 (critic overestimates). TD3 seeds that learned: 23% of actions at the bound. SAC: 0.5% at the bound, 1.7% |u| > 3, Q 81 vs return 82.7.

**Hypothesis verdicts (numbers above).** H1 (tanh saturation locks the actor at the bound): supported (B1; B3 (c) removes collapse at the proxy). H2 (the reward does not favour circling): supported (B2). H3 (short horizon makes circling attractive): not supported (B2: circling is worse than greedy at both discounts; B3 (b) gamma 0.9 does not reduce collapse).

**C3 done** (`results/v2/ws_c/c3_theta_sweep/`, 100 static layouts, theta 0-90 deg in 5 deg steps). Successes: theta 0: 64, 5: 64, 10: 69, 15: 68, 20: 63, 25 (paper): 68, 30: 60, 35: 58, 40: 55, 45: 61, 50: 69, 55: 71, 60: 73, 65: 78, 70: 76, 75: 83, 80: 89, 85: 92, 90: 95. Stalls exist without rotation (36 at theta 0); rotation does not cause them, and only large rotation (>= 65 deg) removes most of them. Diagnostic only: theta stays at 25 deg.

**C4 done; rule applied** (`3825659`): **mom085 selected** (static tuning successes 47 vs apf 30; no held-out dynamic loss). Frozen-set evaluation (`results/v2/ws_c/apfd_eval/`, summary `results/v2/ws_c/summary.json`), APF -> APF-D success %: in distribution 100 -> 98, r = 1.0 94 -> 86, r = 1.58 60 -> 44, noise 5 m/10 deg 94 -> 86, unseen static layouts 68 -> 88, unseen dynamic layouts 98 -> 98, second layout dynamic 96 -> 94. **Negative on the dynamic axes:** damping trades reactivity for stall removal. Reported as a trade-off, not an improvement. Note: `ws_c_apfd_eval.py` wrote one `manifest.json` per run, so the stored manifest covers the last part only; config and seeds are the same for every part.

**D2 done** (`results/v2/ws_d/d2_thresholds/`): 58 SAC collisions on the held-out seeds; d_on = 22.97 m, rho_on = 1.004; the trigger fires in 44.6% of 2 s windows before successful arrivals. Grid d_on {17.97, 22.97, 27.97} x rho_on {0.753, 1.004, 1.255}. Fallback C = ray-APF with momentum 0.85 (C4 selected APF-D).

**WS-A.** `scripts/make_numbers.py` (`results/numbers.json`, `numbers.tex`), `scripts/make_figures.py` (envelope heatmap), `scripts/build_envelope_v2.py` (v2 rows: APF-D now; TD3 fixed, SAC static, arbiters when their evaluations exist).

### 2026-10-13 early: WS-A/E/F/G
**Queues.** b45 (B4+B5 training, ~0.8 episodes/min per run, ~19 h for three waves), d3 (90 jobs at 2 workers, ~19 h), d4 (C-alone and SAC-alone evaluations now; D1/D2 wait for `d3_select` via the new `after` dependency in `scripts/jobqueue.py`), evalb45 (waits for each B4/B5 seed), misc (TD3-static second-layout cell). RAM (about 1 GB free) is the limit.

**Evaluation wrappers checked against stored results** before use: `scripts/ws_b_eval.py` reproduces `collapse.json` for three runs and SAC seed 0's 40/50; its static-layout runs match `layouts_learnedA.json` 15/15. Found and fixed in the wrapper (not in the paper's scripts, which already reset it): `ARCPlanner` caches static geometry across runs, so the cache must be reset per layout.

**WS-F.** `scripts/stats_v2.py`: IQM + stratified bootstrap, Newcombe paired interval (method 10, checked against Newcombe 1998 Table III to 4 decimals), Holm, Wilson. `scripts/ws_c_apfd_compare.py`: APF-D vs APF on seven axes (static layouts +20 points, 95% CI 12 to 28, Holm p = 1.3e-5; dynamic axes all lower, none significant after Holm; r = 1.58: -16 points, CI -30 to 0). APF-D removes 20 of 32 stalls, adds no static failure.

**WS-E.** E1 proposition and proof in `notes/e1_proposition.md` and the paper's new Appendix F. E2 (latency) not run (P2).

**References (WS-G, Section 7).** Added after Crossref lookup: Seto et al. 1998 (Simplex), Phan et al. 2020 (Neural Simplex), Rimon and Koditschek 1992, Ren et al. 2006, Matheron et al. 2020, Newcombe 1998; Bjorck et al. ICLR 2022 (no DOI; content checked on arXiv: tanh saturation in failing runs and a squared pre-tanh penalty, i.e. our B3 (c) is their penalty, cited as such); Holm 1979 (not in Crossref; volume/pages from the JSTOR scan). No paper found that evaluates heading momentum against APF stalls; the text says "we did not find". **[ASK ADI] confirm each new reference.**

**Paper (outside the repo, `../paper.tex`; backups in `../paper_backups/`).** Restructured per WS-G: new abstract and contributions; protocol section (with metrics); planners section with APF-D and the arbiter; results ordered in distribution, matched constraints, collapse mechanism (figure, returns table, 2x2 table), speed ratio with Proposition 1, noise and layouts, transfer (one paragraph), stalls and APF-D (table), arbiter (placeholder), 3D (one paragraph, tables to the appendix), cost; Appendix F proofs. Specific fixes done: "find", pooled 23% (one ARC seed 52%), `
sel` macro for the reward-selected checkpoint, no em dashes, no marketing words, "stable equilibria". Length 45 pages (was 45) after cutting five appendix figures and condensing the introduction, related work, planners and discussion. 18 `\PH{}` placeholders remain (B4, B5, arbiter, IQM, timing).

**WS-F applied to the existing tests (`scripts/ws_f_holm_existing.py`, `results/v2/ws_f/holm_existing/`).** Holm within each family the paper reports changes three claims, all now stated with corrected p in the paper: (1) SAC seeds significantly less reliable than APF: 6 of 10 after Holm, 7 before (seed 2: 0.031 -> 0.125); (2) DWA-CV vs SAC seeds: only seed 9 after Holm (seed 5: 0.022 -> 0.20); (3) collapse count TD3 (moving) vs SAC is not significant after Holm over the three seed-level tests (Fisher 0.033 -> 0.065); the dynamic-success difference holds (permutation 0.004 -> 0.011). Newcombe intervals added to the paired table (DWA and DWA-CV vs APF: -6 points, 95% CI -16 to 2). IQM (bootstrap 95%): TD3 moving 11.8 (0.0-32.8), SAC moving 42.2 (37.7-45.8) of 50.

**WS-A.** `scripts/make_numbers.py` now writes 146 macros; every number in the new sections (collapse, stalls, APF-D table, arbiter thresholds, Holm and Newcombe values) comes from it. The macros caught three rounding slips in hand-typed drafts (-12.65 not -12.66, SE 1.3 not 1.4, Holm 0.62 not 0.63). Older sections still carry numbers typed from `report.md`; converting them is open.

**README** rewritten: six planners, arbiter, how to use the harness (example checked to run).

**Consistency review of the restructured paper** (read-only review pass, findings verified before editing). Fixed: the "disjoint from evaluation sets" claim now states that DWA's weights and APF's gains used the static custom environment; "moving-obstacle training is necessary" softened to "helps but does not suffice" (TD3 static vs moving is not significant); algorithm/collapse claims hedged to the Holm result; DWA-CV noise wording (it degrades faster than APF only at the highest level); APF-D point estimates "lower on five of six dynamic axes, equal on one"; H2 verdict wording; stale appendix pointers (theta sweep, gain sweep results moved into Appendix B, ARC training table); duplicated footnote and cost numbers removed; notation clashes (circle radius a, closing threshold kappa_on, orbit period l, eigenvector xi). Still 45 pages.

### 2026-10-13: Adi's review of C2, E1 and WS-B (follow-ups, in order)
Backup: `../backup_20261013_0118.zip` (paper, figures, results/v2, scripts, notes; no checkpoints), copied to OneDrive `Documents/APF_backups/`. Push to GitHub waits for Adi (repo visibility not confirmed).
1. WS-B: pre-tanh E[u^2] and |u|>3 at step 0, before any gradient update (TD3, 3 seeds): is saturation present at initialisation?
2. C2: the field is not a gradient field at any theta (goal scaling: curl = grad(D^n) x grad(Phi), nonzero off equilibria). Fix note and paper wording.
3. C2: Corollary 1 needs n = 2 and d = 2. Add Corollary 3: tr S < 0 outside obstacles, so div F(q*) < 0 for theta < 90 deg; sink iff det J > 0, saddle iff det J < 0.
4. C2: bias b handled as a shifted equilibrium (F + b = 0); |J^-1 b| per stall; midpoints re-measured from the shifted point.
5. C2 Prop 1: 2-cycles of the linearised map lie on real eigenlines with lambda < 0; transverse multiplier m; check on the 27 cycles (angle to the fast eigenvector, |m| < 1).
6. C2 Prop 2: nearest-saddle distance per sink against turning diameters 38.2 m (cap) and 20.3 m (APF-D); does it predict which stalls each fix removes?
7. WS-B text: H2 as originally stated, marked refuted; mechanism = optimiser trap (saturation + critic overvaluing the bound).
8. E1: speed-independent draw order stated; bit-identity test extended to DWA-CV and noise L2; push-away and separation event counts logged.
9. D3 thresholds untouched (pre-registered); in D4 report where switches occur relative to the goal.

**Follow-up outcomes (2026-10-13).**
1. Step-0 saturation (`scripts/ws_b_init_saturation.py`, `results/v2/ws_b/init_saturation/`, TD3 moving, seeds 10-12): before any update |u| <= 0.035 on a probe of the first 64 buffer states, still < 0.14 after 20 updates; 100% |u| > 3 by update 100-200 in all three seeds, including seed 11, which escaped later. Not an initialisation or input-scaling artefact: training dynamics. The output is then almost state-independent. The earlier "95% at episode 0" was an average over the 540-1370 updates of episode 0.
2-3. Paper and note: field non-conservative at any theta (goal factor); Corollary 1 scoped to the plane; Corollary 3 (tr S < 0) added with proof; all 31 interior equilibria have negative divergence and det J > 0 (sinks).
4. Bias as shifted equilibrium: |b| = 0.011, shift 0.1-2.4 mm; midpoints from the shifted point 0.08-0.85 m.
5. Proposition 1 exact for the linearised map (Appendix F): all 27 node cycles within 3.3 deg of the fast eigenvector, rho 0.03-0.64, |m| < 1 for 26 of 27 (exception: layout 89, most asymmetric cycle).
6. One-mechanism hypothesis not supported: APF-D (turning diameter 20.3 m) removes 20 of 31 interior stalls, the 54 deg/s cap (38.2 m) 15, both 14; nearest-saddle distance AUC 0.72 (cap, p = 0.04) and 0.65 (APF-D, p = 0.19). Reported as measured.
7. H2 as originally stated, marked refuted; collapse framed as an optimiser trap (saturation + critic overvaluing the bound).
8. `scripts/verify_scaleinv_ext.py`: first run (`results/v2/verification/scale_invariance_ext/`, 50 envs): outcomes identical in all 300 compared runs (APF, SAC, DWA-CV; no noise and L2); DWA-CV bit-identical 100/100; SAC bit-identical 96/100 (<= 4 mm); APF: rounding of order 1e-8 m in the obstacle timeline amplified by the field (step count differs by one in 5 of 50 noise-free runs; up to 5.7 m apart under noise). Its separation counter also counted contact-level overlaps that flip with rounding; fixed (material overlaps > 1 um match exactly) and rerun as `scale_invariance_ext2` with a timeline-only check of the goal push-away branch, which no planner run reaches. Draw order stated in the proof.
9. `experiments/arbiter.py` now logs every switch (direction, distance to goal, d_min, v_c, reason); behaviour unchanged (10/10 D3 runs identical to stored results). D3 thresholds untouched. D4 results will report switch locations.
Paper: 45 pages after cutting two hand-built-layout and ARC-training tables, Table A.1 and the custom-3D table (their numbers moved into text) and condensing the contributions, matched-constraints section and conclusion.

**E1 extension, rerun with the fixed counter (`scale_invariance_ext2`, 10 envs) and onset check (`scripts/verify_scaleinv_onset.py`).** Planner runs: events equal at both speeds in every run; push-away fires in 2 runs; outcomes and step counts identical. Long timelines (20,000 ticks, 7 seeds): 4 stay within 3e-7 m at the two speeds (including up to 35 push-away events each); 3 diverge after a contact decided by a near-zero overlap flips (first deviation > 1e-6 m at ticks 2,104, 10,502, 15,918; up to 5 m by 20,000). Floating-point sensitivity of the obstacle-contact dynamics, not a time dependence; no outcome changed in the 300 compared runs. Stated in the paper.

**2026-10-13 11:21: lid closed (sleep).** No reboot; most processes survived and resumed. Four jobs died in the same 8 s with exit 0xC000070A: d3_g8_sac3, d3_g8_sac4, b5_sacstatic_seed6 (training, ep 12) and eval_b5_sacstatic_seed1. Restarted at 13:03: d3 runner rerun for the two D3 jobs (1 at a time); `jobs/b45r.json` retries b5_sacstatic_seed6 from scratch (its partial folder is cleaned); `jobs/evalb45.json` now waits on `b45r/b5_sacstatic_seed6` for that seed, and its idle runner was restarted (retries eval seed 1); the idle d4 runner was restarted with a 0.5 GB RAM floor so the arbiter runs interleave with training instead of waiting for it. The surviving b45 runner and its 7 training runs were left untouched. By then: D3 88/90 done, B4 TD3 seeds 0-3 and B5 SAC seeds 0-2 trained, evaluations done for 6 of them.

### 2026-10-13 evening: D3 done, D4 partial, arbiter handover bug  [ASK ADI]
**D3 selection** (`results/v2/ws_d/d3_selection/`): grid point 6, d_on = 28.0 m, kappa_on = 0.75 (the earliest-switching corner; on the grid edge in both directions, so the optimum may lie outside the pre-registered grid). Mean success over the five held-out sets 0.785, against 0.749-0.779 elsewhere.

**D4, seeds 0-3 (D1) and 0-2 (D2)**: the arbiter beats SAC alone on the hard dynamic axes (r = 1.58: 11-14 vs 5-10; unseen dynamic layouts: 89-91 vs 73-86) but is worse than its fallback alone (APF-D on rays) on every axis (in distribution 43-47 vs 49; r = 1: 24-32 vs 42; noise: 29-32 vs 41) and far below the oracle. Failures (D1, 1800 runs): 227 collisions while C is active, 159 exits from the arena after handing back to SAC, 25 SAC collisions. 16.7 switches and 3.8 chattering switches per run; 36% of the time in C.

**Bug found [ASK ADI].** `experiments/arbiter.py` resets C (APF-D on rays) only at the start of a run: its momentum memory `d_prev` survives between C segments, so at every L -> C handover APF-D blends the new direction with the direction of its last move in an earlier segment, possibly hundreds of metres back. APF-D alone always uses the previous step. DWA-CV is reset at handover, so D2 is not affected by this. This affects D3 (thresholds tuned with the bug) and D4 (D1). An opt-in fix (`handover_reset=True`: momentum starts from the agent's current heading at handover) is in `arbiter.py`; default behaviour unchanged. Held-out diagnostic (SAC seeds 0-1, held-out environment seeds 500000+s, 50 runs each, selected thresholds; not evaluation data): with the bug 46 and 43 successes, with the fix 47 and 45; exits from the arena 9 -> 4, collisions while C is active 2 -> 4. A gain of 3 in 100, within noise at this size; the bug does not explain the gap to APF-D alone (49/50). Options: (a) report D1 as pre-registered and note the bug; (b) fix, rerun D3 (90 jobs, ~10 h) and D4 D1 (10 jobs) as a recorded deviation; (c) fix and rerun D4 only with the current thresholds. Nothing rerun until Adi decides.

## Next
- B4/B5 finish -> evalb45 runs automatically -> fill the 2x2 table, abstract, contributions, discussion, conclusion.
- D3 finishes -> d3_select -> D1/D2 evaluations -> arbiter section and table; title question **[ASK ADI]**.
- Idle timing session: `python scripts/ws_x_timing.py` once nothing else runs (it refuses while training runs).
- WS-A: macros for every number (v2 numbers first); WS-F: Holm within each table's family (existing tables still report unadjusted p).
- Before submission: recheck the RAS guide (abstract limit, highlights; the guide page returns 403 to automated fetches).

### 2026-10-13 night: Adi's decisions
- Title: "Operating Envelopes and Failure Mechanisms of Classical and Learned Path Planners" (paper and README).
- Handover bug: (a) + (c). D1 reported as registered; sensitivity check `d1fix` (handover reset, D3 thresholds unchanged) registered in prereg.md and running (`jobs/d4fix.json`, 10 jobs). Report d1fix against SAC alone and ray-APF-D alone on the same runs (not exact-geometry APF-D), plus d1 against d1fix.
- Contribution 4 rewritten as a negative finding (envelope boundaries per episode do not give per-state switching rules); arbiter removed from the abstract; the lesson goes in the discussion. Switching rate (D1, seeds 0-6): one switch every 13.2-15.0 decisions (about 2.4 s at 6 Hz). The 44.6% trigger rate in success windows is at the D2 median thresholds (23.0 m, 1.00), not the selected ones; recompute at the selected thresholds for the discussion.
- References: all DOI entries rechecked against Crossref; volumes added for the two LNCS chapters (12229, 12397) and the two AAAI-18 papers (32(1)); liu2026 is online-first with no volume yet (recheck before submission). Paper still 45 pages.
- Repo is private (github.com page returns 404 without login): make it public at submission, since the paper links it.
