# Gallery calibration (attn-pool + data-base)

cases=['gbm'] | n_train=2000 epochs=35 SBC 40x40

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
         gbm |   2 |      5.0pp | 2/2       |        56% |        96%
--------------------------------------------------------------------
SBC-pass parameters: 2/2  |  mean calib-err: 5.0pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
           gbm: mu:0.35  sigma:0.18
```
