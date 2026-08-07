# Gallery calibration (attn-pool + data-base)

cases=['double_well'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
 double_well |   3 |      1.6pp | 3/3       |        50% |        89%
--------------------------------------------------------------------
SBC-pass parameters: 3/3  |  mean calib-err: 1.6pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
   double_well: theta1:0.50  theta2:0.74  sigma:0.67
```
