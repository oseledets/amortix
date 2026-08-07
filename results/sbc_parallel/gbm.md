# Gallery calibration (attn-pool + data-base)

cases=['gbm'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
         gbm |   2 |      1.5pp | 1/2       |        50% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 1/2  |  mean calib-err: 1.5pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
           gbm: mu:0.92  sigma:0.00
```
