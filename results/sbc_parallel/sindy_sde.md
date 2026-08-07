# Gallery calibration (attn-pool + data-base)

cases=['sindy_sde'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
   sindy_sde |   5 |      1.3pp | 5/5       |        48% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 5/5  |  mean calib-err: 1.3pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
     sindy_sde: c0:0.16  c1:0.29  c2:0.14  c3:0.16  sigma:0.85
```
