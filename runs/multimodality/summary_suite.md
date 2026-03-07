# Multimodal Summary Suite

| Run | Step (s) | Gen (s) | Ref (s) | Old Log Prob (s) | Update Actor (s) | Update Weights (s) | Sync Rollout Weights (s) | Payload (MB) | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_step_off_disaggregate_3b_1node | 20.21 | 11.52 | 5.45 | - | 1.64 | 1.58 | 0.84 | 18.55 | 104.85 |
| one_step_off_disaggregate_3b_8node32gpu | 52.17 | 13.07 | 3.84 | - | 9.68 | 25.17 | 12.73 | 207.37 | 44.21 |
| one_step_off_disaggregate_3b_smoke2 | 26.76 | 16.65 | 6.90 | - | 1.62 | 1.56 | 0.81 | 33.95 | 91.46 |
| one_step_off_disaggregate_7b_1node | 32.20 | 14.42 | 12.62 | - | 3.32 | 1.84 | 0.87 | 18.38 | 80.33 |
| sync_colocate_3b_1node | 26.65 | 16.66 | 2.99 | 2.62 | 1.51 | 2.87 | - | 23.00 | 46.97 |
| sync_colocate_3b_8node32gpu | 27.57 | 6.32 | 1.89 | 4.57 | 6.03 | 8.75 | - | 84.86 | 19.20 |
| sync_colocate_7b_1node | 30.36 | 16.66 | 6.29 | 2.10 | 1.87 | 3.44 | - | 27.08 | 41.64 |
