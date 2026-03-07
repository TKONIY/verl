# Component Breakdown

> `transport_proxy` is an inference, not a direct network timer. It uses `timing_s/update_weights` plus `timing_s/sync_rollout_weights` as the lower-bound time spent on cross-rank parameter / state movement and associated synchronization.

| Run | Step (s) | Gen % | Ref % | Old Log Prob % | Update Actor % | Update Weights % | Sync Rollout % | Transport Proxy % | Payload MB | Payload / Sample MB | Overlap Lower Bound % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sync 3B 1-node | 26.65 | 62.5 | 11.2 | 9.8 | 5.7 | 10.8 | 0.0 | 10.8 | 23.00 | 2.876 | 0.0 |
| one_step_off 3B 1-node | 20.21 | 57.0 | 27.0 | 0.0 | 8.1 | 7.8 | 4.2 | 12.0 | 18.55 | 2.319 | 4.1 |
| sync 3B 32-GPU | 27.57 | 22.9 | 6.8 | 16.6 | 21.9 | 31.7 | 0.0 | 31.7 | 84.86 | 2.652 | 0.0 |
| one_step_off 3B 32-GPU | 52.17 | 25.1 | 7.4 | 0.0 | 18.6 | 48.3 | 24.4 | 72.6 | 207.37 | 3.240 | 23.6 |
| sync 7B 1-node | 30.36 | 54.9 | 20.7 | 6.9 | 6.2 | 11.3 | 0.0 | 11.3 | 27.08 | 3.385 | 0.0 |
| one_step_off 7B 1-node | 32.20 | 44.8 | 39.2 | 0.0 | 10.3 | 5.7 | 2.7 | 8.4 | 18.38 | 2.297 | 2.7 |
