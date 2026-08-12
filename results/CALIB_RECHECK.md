# Gallery calibration (attn-pool + data-base)

cases=['stoch_lv', 'sindy_sde'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
    stoch_lv |   4 |      1.3pp | 3/4       |        50% |        89%
   sindy_sde |   5 |      1.2pp | 4/5       |        49% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 7/9  |  mean calib-err: 1.2pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
      stoch_lv: alpha:0.00  beta:0.13  delta:0.05  gamma:0.97
     sindy_sde: c0:0.59  c1:0.19  c2:0.24  c3:0.18  sigma:0.00
```
