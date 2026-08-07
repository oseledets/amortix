# Gallery calibration (attn-pool + data-base)

cases=['stoch_lv'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
    stoch_lv |   4 |      1.4pp | 3/4       |        51% |        89%
--------------------------------------------------------------------
SBC-pass parameters: 3/4  |  mean calib-err: 1.4pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
      stoch_lv: alpha:0.00  beta:0.21  delta:0.52  gamma:0.19
```
