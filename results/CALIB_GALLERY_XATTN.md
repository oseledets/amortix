# Gallery calibration (attn-pool + data-base)

cases=['ou', 'seir', 'gbm', 'cir', 'double_well', 'stoch_lv', 'fhn', 'sindy_sde'] | n_train=50000 epochs=60 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
          ou |   3 |      7.6pp | 2/3       |        63% |        92%
        seir |   5 |      2.8pp | 3/5       |        48% |        88%
         gbm |   2 |      1.6pp | 2/2       |        53% |        92%
         cir |   3 |      7.5pp | 1/3       |        61% |        91%
 double_well |   3 |      2.4pp | 1/3       |        48% |        89%
    stoch_lv |   4 |      4.3pp | 1/4       |        45% |        86%
         fhn |   4 |      1.9pp | 3/4       |        48% |        88%
   sindy_sde |   5 |      1.3pp | 4/5       |        49% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 17/29  |  mean calib-err: 3.7pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
            ou: theta:0.11  mu:0.00  sigma:0.30
          seir: beta1:0.12  beta2:0.00  alpha:0.57  gamma_r:0.75  gamma_d:0.00
           gbm: mu:0.19  sigma:0.15
           cir: a:0.01  b:0.00  sigma:0.61
   double_well: theta1:0.40  theta2:0.00  sigma:0.00
      stoch_lv: alpha:0.00  beta:0.00  delta:0.06  gamma:0.04
           fhn: a:0.22  b:0.24  eps:0.08  I:0.00
     sindy_sde: c0:0.39  c1:0.64  c2:0.21  c3:0.23  sigma:0.00
```
