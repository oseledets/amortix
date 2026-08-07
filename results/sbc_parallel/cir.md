# Gallery calibration (attn-pool + data-base)

cases=['cir'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
         cir |   3 |      0.9pp | 3/3       |        50% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 3/3  |  mean calib-err: 0.9pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
           cir: a:0.34  b:0.42  sigma:0.65
```
