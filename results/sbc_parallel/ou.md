# Gallery calibration (attn-pool + data-base)

cases=['ou'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
          ou |   2 |      1.5pp | 2/2       |        52% |        92%
--------------------------------------------------------------------
SBC-pass parameters: 2/2  |  mean calib-err: 1.5pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
            ou: theta:0.87  sigma:0.32
```
