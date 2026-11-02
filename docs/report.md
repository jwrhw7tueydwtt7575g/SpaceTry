# Runtime assurance for learned spacecraft docking controllers

Project report for SpaceForge. Implementation details are in [design.md](design.md) and measured
numbers in [results.md](results.md).

## 1. Problem

A chaser spacecraft starts within 20 m of a target in low Earth orbit and must dock: reach the
target within 0.5 m at a relative speed under 0.1 m/s. Inside a 5 m keep-out zone its closing speed
must stay under 0.2 m/s. There is no drag, so every bit of speed gained has to be removed with
thrust, and relative motion in orbit couples the axes: a radial offset produces along-track drift.

Formally, with relative state `X = [r, v] ∈ ℝ⁶` in the target's Hill frame, mean motion
`n = 0.00113 rad/s` and commanded acceleration `u ∈ [−1, 1]³ × 0.1 m/s²`, find a controller such
that for some `t_f ≤ 300 s`:

```
‖r(t_f)‖ < 0.5 m   and   ‖v(t_f)‖ < 0.1 m/s
```

and the closing speed `−v·r/‖r‖` stays under a range-dependent limit along the way.

Learned controllers, such as a PPO policy, can fly this problem, but nothing guarantees how they
behave at a state they were not trained on. The project asks whether a simple runtime filter can
let such a controller fly while keeping the approach within the speed limit.

## 2. Related work

| Ref | Work | Relevance here |
|---|---|---|
| [1] | Clohessy and Wiltshire, 1960 | Linearized relative motion about a circular orbit; closed-form solution used for propagation |
| [2] | Hablani, Tapper and Dana-Bashian, 2002 | Glideslope guidance for rendezvous, built on the CW solution; the basis of the backup controller |
| [3] | Schulman et al., 2017 | PPO, the algorithm used to train the learned controller |
| [4] | Seto, Krogh, Sha and Chutinan, 1998 | The Simplex architecture: an unverified controller runs under a monitor that switches to a verified one |
| [5] | Ames et al., 2019 | Control barrier functions, a related way to enforce safe sets by filtering control inputs |
| [6] | Gaudet, Linares and Furfaro, 2020 | PPO for 6-DOF powered planetary landing; an example of learned guidance in a related setting |

## 3. Method

**Dynamics.** The CW equations are propagated one step at a time with their exact state transition
matrix, holding the commanded acceleration constant over the step:

```
X(k+1) = Φ(dt) X(k) + Γ(dt) u(k)
```

The environment is a Gymnasium environment with a 10-value observation (scaled position and
velocity, fuel, elapsed time, bearing) and a 3-value action. Episodes end on docking, collision
(passing through the 0.5 m sphere at or above 0.1 m/s), empty tank, leaving a 100 m volume, or after
300 steps.

**Controllers.**

- *Glideslope.* Commands a closing speed of `min(0.5, d/30, 0.5·sqrt(2·a_max·d))` toward the target,
  tracks it with a proportional gain, and cancels the CW drift terms in feed-forward.
- *PPO.* A 64-64 MLP trained with Stable-Baselines3 for 300k steps on random starts. The shipped
  snapshot was chosen on held-out seeds.
- *Reckless.* Full thrust at the target, used only to test the filters.

**Filters.**

- *v1* zeroes any command inside 5 m that does not reduce speed while the vehicle exceeds
  0.2 m/s.
- *v2* is a Simplex filter. Each step it predicts the controller's command and checks eight points
  through the step against a closing-speed limit:

```
r ≥ 5 m:  v_lim(r) = sqrt(0.2² + 2 · 0.5 · a_max · (r − 5))
r < 5 m:  v_lim(r) = clamp(0.04 · r, 0.05, 0.2)
```

If the prediction violates the limit, the glideslope's command is used for that step. If that would
also violate it, a brake sized to the excess closing speed is used. The outer curve guarantees that
a vehicle below it can reach the 5 m boundary at 0.2 m/s with half the available thrust.

**Evaluation.** Each controller-filter pair is run from the same random starts. Each run records
whether it docked, whether it collided, fuel, Δv, filter interventions, and the closing speed when
it crossed into the 5 m zone. A browser version of the simulator, checked against the Python code
to 1e-9 m, is used for interactive runs and the reckless-law Monte Carlo.

## 4. Results

**Recorded runs** (seed 42 for the docking runs, seed 7 for the reckless runs; 0.1 m/s² authority):

| | Glideslope | PPO | Reckless, no filter | Reckless, v1 | Reckless, v2 |
|---|---|---|---|---|---|
| Filter | v1 | v1 | off | v1 | v2 |
| Start range | 14.33 m | 18.22 m | 6.00 m | 6.00 m | 6.00 m |
| Steps | 98 | 208 | 11 | 14 | 72 |
| Δv | 1.381 m/s | 2.148 m/s | 1.106 m/s | 0.500 m/s | 0.595 m/s |
| Filter actions | 0 | 0 | 0 | 9 | 69 |
| Outcome | docked | docked | hit at 1.10 m/s | hit at 0.50 m/s | docked at 0.017 m/s |

**Over many starts:**

- The glideslope docks 200/200 random starts in 107 steps on average, and the filter never acts.
- The PPO policy docks 500/500 fresh starts with every filter setting. Unfiltered, it enters the
  keep-out zone at up to 0.958 m/s, almost five times the limit. With v2 the maximum is 0.194 m/s,
  at the cost of 10.8 interventions per flight.
- Against the reckless law over 250 close starts, collisions drop from 247 (no filter) to 137 (v1)
  to 0 (v2).

**Training.** At 1 m/s² of authority per unit command, PPO did not learn to dock at all (0%). At
0.1 m/s² it reached 100% with the same settings. At the higher authority, exploration noise alone
changes the speed by about 1 m/s per step against a 0.1 m/s tolerance. Success across snapshots of
one run varied between 1% and 100%.

## 5. Discussion

The veto filter shows why removing thrust is not enough near a target in orbit. The vehicle keeps
its speed, so v1 halves the impact speed but does not prevent the impact. Substituting a known
controller does prevent it.

The more useful finding came from plotting closing speed against range. The first version of v2
only applied its limit inside 5 m, and the PPO policy, which docks every time, was arriving at the
boundary at almost 1 m/s. Docking success alone would never have shown this. Extending the limit
outward as a braking curve fixed it without changing the glideslope's behaviour.

Limitations:

- The PPO policy is slower and uses more fuel than the glideslope, and its training is unstable.
- It was trained and tested only without randomization and faults. The glideslope was tested with
  both (120/120 at each level).
- The ROS mission controller does not yet use the outer braking curve.
- Closed-loop docking in Gazebo has not been run.

## 6. Conclusion

A Simplex-style filter with a glideslope backup lets an unverified controller fly a docking
approach while holding a closing-speed limit along the whole approach, not just at the end. In
these tests the filter never had to act on the glideslope, kept a trained PPO policy under the
limit at a cost of about 11 interventions per flight, and turned a reckless controller's 247
collisions out of 250 into 250 dockings.

## References

1. W. H. Clohessy and R. S. Wiltshire, "Terminal guidance system for satellite rendezvous,"
   *Journal of the Aerospace Sciences*, vol. 27, no. 9, pp. 653-658, 1960. doi:10.2514/8.8704
2. H. B. Hablani, M. L. Tapper and D. J. Dana-Bashian, "Guidance and relative navigation for
   autonomous rendezvous in a circular orbit," *Journal of Guidance, Control, and Dynamics*,
   vol. 25, no. 3, pp. 553-562, 2002. doi:10.2514/2.4916
3. J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, "Proximal policy optimization
   algorithms," arXiv:1707.06347, 2017.
4. D. Seto, B. Krogh, L. Sha and A. Chutinan, "The Simplex architecture for safe online control
   system upgrades," in *Proc. American Control Conference*, pp. 3504-3508, 1998.
   doi:10.1109/ACC.1998.703255
5. A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath and P. Tabuada, "Control barrier
   functions: Theory and applications," in *Proc. European Control Conference*, pp. 3420-3431,
   2019. doi:10.23919/ECC.2019.8796030
6. B. Gaudet, R. Linares and R. Furfaro, "Deep reinforcement learning for six degree-of-freedom
   planetary landing," *Advances in Space Research*, vol. 65, no. 7, pp. 1723-1741, 2020.
   doi:10.1016/j.asr.2019.12.030
