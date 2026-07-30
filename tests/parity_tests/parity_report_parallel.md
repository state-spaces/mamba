# Parallel ANE Parity Report: StatefulMambaParallelHybrid

Generated: 2026-05-11 15:53:41
Config: d_model=256, d_state=64, headdim=64, seq_length=256, num_classes=10
Sequence: 64 independent inputs, seed=42, compute_units=CPU_AND_NE

## MPS_FP16: `OG_FP32 (golden)` → `Portable_FP16 (mps)`
Tolerances: max_abs < 0.01, cosine_sim > 0.9999

| Metric | Value | Status |
|--------|-------|--------|
| max_abs (worst frame)     | 3.628e-05    | **PASS** |
| mean_abs_avg              | 8.629e-06 |  |
| cosine_sim_min            | 1.000000 |  |

### max_abs distribution across measurement frames
```
[1.07e-05,1.39e-05)    2  ###
[1.39e-05,1.71e-05)    1  #
[1.71e-05,2.03e-05)   11  #################
[2.03e-05,2.35e-05)   19  ##############################
[2.35e-05,2.67e-05)   14  ######################
[2.67e-05,2.99e-05)    9  ##############
[2.99e-05,3.31e-05)    7  ###########
[3.31e-05,3.63e-05)    1  #
```

## CoreML_ANE: `OG_FP32 (golden)` → `CoreML CPU_AND_NE`
Tolerances: max_abs < 0.03, cosine_sim > 0.999

| Metric | Value | Status |
|--------|-------|--------|
| max_abs (worst frame)     | 5.780e-04    | **PASS** |
| mean_abs_avg              | 1.467e-04 |  |
| cosine_sim_min            | 0.999973 |  |

### max_abs distribution across measurement frames
```
[1.87e-04,2.35e-04)    9  ##############
[2.35e-04,2.84e-04)   11  #################
[2.84e-04,3.33e-04)   19  ##############################
[3.33e-04,3.82e-04)   10  ###############
[3.82e-04,4.31e-04)    5  #######
[4.31e-04,4.80e-04)    5  #######
[4.80e-04,5.29e-04)    4  ######
[5.29e-04,5.78e-04)    1  #
```

