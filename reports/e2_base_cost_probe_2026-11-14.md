# E2 baseline cost probe

The training-only probe passed, including exact CUDA checkpoint continuation for
both learned arms. It consumed **23.489078 GPU seconds** of E0's separate
1,800-second allowance. Total measured cost was 59.187156383 CPU seconds and
72.494549913 elapsed seconds. The 24 GPU-hour SL pilot had not started.

The [configuration](../configs/e2_throughput_v1.json),
[command](../docs/baseline_pilot.md), and
[evidence manifest](../manifests/e2_throughput_v1.json) bind the measurements.
The private [W&B run](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/102c6647d4684aa58bf8b59e5e5b9999)
retains source/configuration references. Source was commit `5857e0b` plus
patch `6a81dd629e607f3d56c3fc249b4a80a08feef1e2803c2444e56932917d4fd2a0`.
The evidence retains the exact probe supervisor source. Later pre-launch fixes
to progress-file scanning do not change the measured probe or its source patch.

## Measured work

Five alternating repetitions used the same 8-block, 128-channel float32 model
with and without rank conditioning. Each case included two warmup and ten timed
updates. Both exact-resume checks passed. All 246 actual updates and 23,424
presentations used training data. Probe checkpoints cannot initialize the pilot.

| Arm | Batch | Examples/second, minimum / median / maximum | Peak allocated tensor memory |
| --- | ---: | ---: | ---: |
| Conditioned | 64 | 2474.26 / 2581.89 / 2583.96 | 0.495 GiB |
| Unconditioned | 64 | 2584.96 / 2586.37 / 2592.30 | 0.495 GiB |
| Conditioned | 128 | 2896.84 / 2897.80 / 2911.01 | 0.912 GiB |
| Unconditioned | 128 | 2885.89 / 2893.49 / 2903.82 | 0.912 GiB |

The RTX 3090 reported driver 595.99.02, 24,576 MiB total memory, 38 MiB in use
and 1% utilization before CUDA initialization. Python was 3.12.13 and PyTorch
2.15.0+cu132. Deterministic algorithms were enabled; TF32 and cuDNN benchmarking
were disabled. Timed work includes batching/transfer, forward/backward, optimizer
update, metric extraction and synchronization. Loading, model construction,
warmup and checkpoint work remain inside the charged allocation.

Each learned arm made 25 timed CPU forward calls and five warmups on five fixed
training positions. Median calls were 15.944 ms and 15.951 ms using prepared
observations, excluding replay/encoding. Human SL made 25 timed requests, five
warmups and five version requests, restarting its native process for every
repetition. Its median was 341.969 ms and maximum 392.118 ms, including history
validation and trace persistence. These different boundaries are not an
inference-speed comparison. Native neural-forward counts remain unknown.

All phases completed: CUDA training used 23.4891 elapsed seconds, CPU inference
11.4577, and the CPU reference 33.7076. Total cost includes wrapper/tracking
time. CPU accounting includes the caller and charged children; tracking-service
CPU is not fully measured. No held-out score, quality result or complete-game
throughput is established by this probe.

## Frozen next comparison

The [pilot plan](../configs/e2_base_plan_v1.json) chooses batch 128 and 20 full
epochs per arm: 3,600 updates and 458,320 presentations. It uses a paired seed,
identical initial tensors/data order, Adam at 0.001, and the final epoch-20
checkpoint without validation-based early stopping. The measured minimum
suggests about 159 seconds of update work per arm; each job reserves up to
1,200 seconds plus 10 for shutdown. This is an estimate, not measured pilot cost.

The two development roles remain separate: 14 population-development games with
3,363 positions, and 10 unseen-account validation-development games with 2,563.
Their observed-rank aggregates were declared before predictions. The all-16-rank
aggregate is unavailable, every cell lacks promotion support, and the roles
have only 19 and 11 accounts. Their intervals cannot pass E0's account-count
gate. Confirmation and final targets remain sealed.

The pilot also declares an unknown-rank diagnostic, 32 raw samples per position,
and 12 color-balanced raw games against two weak policies. This is exploratory
work, not a full validation panel or raw-profile qualification. Unresolved and
unfilled games remain in the denominator.
