# GPU Parity Report: StatefulMambaHybrid1D

Generated: 2026-05-07 08:31:15
Config: d_model=512, d_state=64, expand=1, headdim=64, num_heads=8, seq_length=224
Sequence: warmup=32 + measure=64 frames, seed=42, batch=1

## IMPL: `OG_FP32 (Mamba3 CUDA)` → `ANE_FP32 (MambaBlock)`
Tolerances: max_abs < 0.01, cosine_sim > 0.999

| Metric | Value | Status |
|--------|-------|--------|
| max_abs (worst frame)     | 6.006e-06    | **PASS** |
| mean_abs_avg              | 5.389e-07 |  |
| cosine_sim_min            | 1.000000 |  |

### max_abs distribution across measurement frames
```
[6.44e-07,1.31e-06)   14  ######################
[1.31e-06,1.98e-06)   19  ##############################
[1.98e-06,2.65e-06)   15  #######################
[2.65e-06,3.32e-06)    5  #######
[3.32e-06,4.00e-06)    3  ####
[4.00e-06,4.67e-06)    4  ######
[4.67e-06,5.34e-06)    3  ####
[5.34e-06,6.01e-06)    1  #
```

## PREC: `ANE_FP32` → `ANE_FP16`
Tolerances: max_abs < 0.01, cosine_sim > 0.999

| Metric | Value | Status |
|--------|-------|--------|
| max_abs (worst frame)     | 3.046e-05    | **PASS** |
| mean_abs_avg              | 5.252e-06 |  |
| cosine_sim_min            | 1.000000 |  |

### max_abs distribution across measurement frames
```
[2.74e-05,2.78e-05)    5  #######
[2.78e-05,2.82e-05)    3  ####
[2.82e-05,2.85e-05)    4  ######
[2.85e-05,2.89e-05)   11  #################
[2.89e-05,2.93e-05)   19  ##############################
[2.93e-05,2.97e-05)    9  ##############
[2.97e-05,3.01e-05)    8  ############
[3.01e-05,3.05e-05)    5  #######
```

