# Gallery calibration (attn-pool + data-base)

cases=['fhn'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
         fhn |   4 |      1.5pp | 4/4       |        50% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 4/4  |  mean calib-err: 1.5pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
           fhn: a:0.76  b:0.89  eps:0.66  I:0.89
```
