# ANE Parity Report: StatefulMambaHybrid1D

Generated: 2026-05-07 10:32:39
Config: d_model=512, d_state=64, headdim=64, num_heads=8, seq_length=224
Sequence: warmup=32 + measure=64 frames, seed=42, compute_units=CPU_AND_NE

## ANE: `pytorch FP32 (CPU)` → `CoreML CPU_AND_NE`
Tolerances: max_abs < 0.03, cosine_sim > 0.999

| Metric | Value | Status |
|--------|-------|--------|
| max_abs (worst frame)     | 2.102e-04    | **PASS** |
| mean_abs_avg              | 2.067e-05 |  |
| cosine_sim_min            | 0.999998 |  |

### max_abs distribution across measurement frames
```
[3.35e-05,5.56e-05)   15  ############################
[5.56e-05,7.77e-05)   14  ##########################
[7.77e-05,9.98e-05)   12  ######################
[9.98e-05,1.22e-04)   16  ##############################
[1.22e-04,1.44e-04)    2  ###
[1.44e-04,1.66e-04)    3  #####
[1.66e-04,1.88e-04)    1  #
[1.88e-04,2.10e-04)    1  #
```

