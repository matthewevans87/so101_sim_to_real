# SKRL Notes


Here's how all these values connect, working from the physics level up:

---

## From physics to policy steps

From baseline.yaml:
- `sim.dt = 1/120 s` — physics engine ticks at **120 Hz**
- `decimation = 2` — the policy only sees/acts every 2 physics steps

So the **policy step rate** is `120 / 2 = 60 Hz`, meaning `step_dt = 1/60 s ≈ 16.7 ms` — one action is held for two physics sub-steps.

---

## Episode length in steps

```
episode_length_s = 10.0 s
steps_per_episode = 10.0 / (1/60) = 600 policy steps
```

Each episode (from reset to done) lasts **600 policy steps**.

---

## PPO rollout collection

`rollouts: 32` is the number of policy steps collected (per environment) before each gradient update. With `num_envs=4`:

```
data collected per PPO update = 32 steps × 4 envs = 128 transitions
```

Since `rollouts (32) << steps_per_episode (600)`, each rollout buffer is a **fragment of an episode**, not a full trajectory. Episodes span across multiple rollouts. The episodes continue running in the background through resets; skrl doesn't wait for episode boundaries.

---

## PPO gradient update

After collecting 128 transitions, PPO optimizes the policy:

```
mini_batches: 8   → each mini-batch = 128 / 8 = 16 samples
learning_epochs: 8 → entire batch reshuffled and iterated 8 times
→ total gradient steps per PPO update = 8 epochs × 8 mini-batches = 64
```

---

## Timesteps and max_iterations

skrl's `trainer.timesteps` counts individual `env.step()` calls (each call steps all `num_envs` in parallel). A PPO update fires every `rollouts` timesteps:

```
max_iterations = timesteps / rollouts
→ timesteps = max_iterations × rollouts
```

With `--max-iterations 100`: `timesteps = 100 × 32 = 3200`. That's why the display shows **3200** — it's the total number of `env.step()` calls, equaling 100 PPO gradient update cycles.

The YAML's `timesteps: 100000000` is the default for a full training run: `100M / 32 ≈ 3.1M PPO updates`.

---

## Summary table (with your baseline values)

| Parameter                   | Value           | Meaning                               |
| --------------------------- | --------------- | ------------------------------------- |
| `sim.dt`                    | 1/120 s         | Physics tick                          |
| `decimation`                | 2               | Physics steps per policy step         |
| `step_dt`                   | 1/60 s          | Real time per policy action           |
| `episode_length_s`          | 10.0 s          | Max episode wall time                 |
| `steps_per_episode`         | **600**         | Policy steps before forced reset      |
| `rollouts`                  | 32              | Steps collected per env before update |
| `num_envs`                  | 4               | Parallel envs                         |
| `transitions_per_update`    | **128**         | `rollouts × num_envs`                 |
| `mini_batches`              | 8               | Chunks per epoch (16 samples each)    |
| `learning_epochs`           | 8               | Passes over the 128 transitions       |
| `gradient_steps_per_update` | **64**          | `epochs × mini_batches`               |
| `--max-iterations 100`      | 100 PPO updates | = `timesteps / rollouts = 3200 / 32`  |