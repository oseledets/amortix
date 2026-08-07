# Gallery calibration (attn-pool + data-base)

cases=['linear_gaussian'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
linear_gaussian |   4 |      0.8pp | 3/4       |        50% |        91%
--------------------------------------------------------------------
SBC-pass parameters: 3/4  |  mean calib-err: 0.8pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
  linear_gaussian: m1:0.44  m2:0.03  m3:0.57  m4:0.06
```
