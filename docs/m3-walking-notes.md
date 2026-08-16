# M3 — Walking policy: investigation notes

Status: **parked**. The pipeline is not yet capable of cold-start humanoid gait
discovery. This doc records what was tried, the blockers found, and the retry
checklist so the next attempt starts from knowledge instead of scratch.

## What was tried

| Run | Setup | Result |
|---|---|---|
| 1-2 | Original reward (velocity×heading), pre-standardization | Plateaued at standing-still (also ran on a broken action space) |
| 3 | Standardized Isaac Lab terms, `min_height: 0.9`, 2M steps | **Learned nothing** — every episode terminated on step 1 |
| 4 | Reward reshaped (alive 0.1, forward 4.0), `min_height: 0.9` | Same as run 3 |
| 5 | `min_height: 0.3` + reshaped reward | 123-step sink episodes, no forward motion |
| 6 | `min_height: 0.45` (kill the slow sink) | 87-step sink episodes, still no motion |
| 7 | + height & progress rewards (`prev_observation` task state) | Same 87-step sink; reward/step rose (signal flows) but behavior unchanged |

All runs: `agent=humanoid` (berkeley menagerie, 37-obs/12-act), single-env SB3 PPO,
CPU (GPU measured 3.6× *slower* single-env), checkpoints every 500k.

## Blocker chain (in order of discovery)

1. **`min_height` was calibrated for a deleted robot.** The berkeley humanoid's
   root height at reset is **0.6 m**; the config said 0.9 → instant termination
   → reward always 0 → zero learning. The threshold is now 0.45 (standing 0.6,
   fallen torso ~0.1).
2. **Constant-action attractor.** The policy converges to a *fixed* action
   vector (`|a| ≈ 0.7` every step) — no feedback control. The robot then
   passively sinks and falls. `ctrl=0` is worse (position actuators slam
   joints to 0 and launch the robot), so "constant push" beats "do nothing".
3. **Exploration collapse.** The policy's action `std` → **0** by ~500k steps.
   With entropy gone, PPO can no longer search — everything after is frozen.
   `ent_coef 0.03` was insufficient.
4. **Single-env experience thinness.** One trajectory per update, mostly from
   the same start pose (SB3 resets without seeding). No diversity → no
   pressure to generalize → the constant-action sink is a stable fixed point.
5. **GPU is net-negative single-env** (239 vs 856 steps/s): per-step tensor
   transfer overhead swamps tiny MLPs. GPU only pays off with batched
   (vectorized) rollouts.

## The current task recipe (already committed)

`configs/task/walk_forward.yaml`: `min_height: 0.45`, `forward_coef: 4.0`,
`heading_coef: 0.8`, `upright_coef: 0.3`, `alive_reward: 0.1`,
`action_rate_coef: 0.01`, `progress_coef: 2.0` (per-step Δx, uses
`prev_observation`), `height_coef: 2.0` (fights the slow sink).
`sim.initial_noise: 0.05` randomizes start poses so "stand still at the default
pose" is not memorizable.

## Retry checklist (when the pipeline is ready)

- [ ] **Vectorized envs landed** (`algo.n_envs` via `DummyVecEnv`) — 8×
      experience diversity per update and batched inference
      (measured: humanoid 8000 steps — single 22s, DummyVecEnv×8 30.5s,
      SubprocVecEnv×8 77s. The sim is cheap (~0.05 ms/step) and the policy
      (~1 ms) is the bottleneck, so in-process vectorization wins over
      multi-process: SubprocVecEnv's per-step IPC overhead dwarfs the sim.
      Diversity, not parallelism, is the value of vectorizing here)
- [ ] **Keep exploration alive** — try `ent_coef 0.1+`, or a larger net; verify
      the policy's action std does not collapse (checkpoint at ~500k)
- [ ] **Try RLtools as a second engine** — built for fast humanoid training;
      a different optimizer landscape may escape the constant-action attractor
- [ ] Verify on the cartpole toy first: `n_envs=8` must still solve balance,
      then scale to the humanoid
- [ ] GPU note: currently OS-blocked on this laptop ("GPU access blocked by the
      operating system") — check `nvidia-smi` after a reboot before GPU runs

## Infrastructure built along the way (useful for the retry)

- Checkpoints every N steps with sidecar manifests (`algo.checkpoint_freq`)
- Mid-run `weir-eval` on periodic checkpoints (manifest callback)
- `prev_observation` in the task protocol (progress rewards)
- `sim.initial_noise` config
- `device: cpu|cuda|auto` from config; optional GPU torch via `uv sync --extra gpu`
