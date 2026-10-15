# WS-E, E1: outcomes depend on speeds only through r

Draft for Section 6.6 (the speed-ratio envelope). Check: `notes/scale_invariance_audit.md`
(Section 3.4 audit and bit-identity test). **[ASK ADI]** check together with the C2 note.

## Setting

Agent ground speed v; obstacle i moves with velocity c_i v_max u_i, where u_i is a unit vector,
0 < c_i <= 1 and (c_i, u_i(0)) are drawn from the motion seed independently of any speed, and
v_max = 4.75 m/s is the fastest obstacle speed. r = v_max / v.

Every planner decides at a fixed step length per decision (APF and DWA: Delta s = 1.8 m;
learned planners: 3.0 m), so its decision period is Delta s / v. The obstacle timeline advances in
ticks of Delta t = (1.8 m) / (3 v), so an APF or DWA decision spans 3 ticks and a learned decision 5.

## Proposition

Fix the environment (layout and motion seed), the planner, its seeds and its decision and time
budgets (counted in decisions). Then the sequence of agent positions, obstacle positions, and hence
the outcome (success, collision, timeout) and every path metric, depend on (v, v_max) only through r.

## Proof

Induction over obstacle ticks. Write the state after tick k as the agent position, the obstacle
positions and the obstacle velocities divided by v.

1. *Obstacle motion.* In one tick obstacle i moves by c_i v_max u_i Delta t = 0.6 m x c_i r u_i: a
   function of r and the state only. The goal push-away is a positional displacement per tick;
   wall reflection, obstacle-obstacle separation and the velocity exchange act on positions and
   reverse or swap the direction u_i while restoring each speed to its pre-contact magnitude, so
   velocity / v stays a function of r and the state.
2. *Planner decisions.* Each planner maps an observation measured in metres (obstacle positions,
   ray ranges, clearances) to a displacement in metres:
   - APF: 1.8 m along the field direction; stall jitter, escape steps and push-outs are in metres
     per decision.
   - Learned planners: 3.0 m along a heading that changes by at most 9 deg per decision; the
     observation history is counted in decisions.
   - DWA: its velocity, turn-rate and acceleration limits are set proportional to v, v and v^2, so
     the admissible displacements per decision (turning radius 11.5 m, braking distance 18 m,
     look-ahead 54 m) are the same at every v. DWA-CV estimates obstacle velocity as displacement
     per decision, i.e. velocity x (Delta s / v), a function of r and the state.
   - Sensing noise is drawn once per decision in metres and degrees.
3. *Termination.* Collision, goal and arena tests are positional; budgets are counted in decisions.

So the state after tick k + 1 is a function of the state after tick k and r alone, and the
initial state does not depend on any speed. By induction the whole run is a function of r.  QED

## Remarks

- In the implementation the claim holds to floating-point tolerance: with r held fixed and both
  speeds halved, SAC's runs are bit-identical (10/10) and APF's have the same outcome and step
  count (10/10) with paths within 4.8 cm, from rounding of order 1e-8 m in the obstacle timeline
  (`results/v2/verification/scale_invariance/`).
- Consequence: varying obstacle speed at fixed agent speed and varying agent speed at fixed
  obstacle speeds give the same curve; the speed sweep is a sweep in r, and a separate
  obstacle-speed experiment cannot add information.
- What breaks it: any quantity carried in absolute time that is not tied to the decision period,
  e.g. a fixed perception-to-action latency tau (outcomes would then also depend on v tau / Delta s,
  the latency in steps), actuator dynamics with fixed time constants, or a wall-clock decision
  deadline. The paper's limitations section lists latency; E2 would measure it.
- Not covered: moving-obstacle training (always at r = 0.26) and the 3D worlds.
