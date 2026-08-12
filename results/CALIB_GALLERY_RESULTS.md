# Gallery calibration (attn-pool + data-base)

cases=['linear_gaussian', 'ou', 'seir', 'gbm', 'cir', 'double_well', 'stoch_lv', 'fhn', 'sindy_sde'] | n_train=40000 epochs=35 SBC 500x200

```

        case | dim | calib-err |  SBC pass | mean cov50 | mean cov90
--------------------------------------------------------------------
linear_gaussian |   4 |      0.9pp | 4/4       |        51% |        91%
          ou |   2 |      1.0pp | 2/2       |        52% |        90%
        seir |   5 |      1.6pp | 5/5       |        49% |        89%
         gbm |   2 |      1.2pp | 2/2       |        50% |        90%
         cir |   3 |      1.0pp | 3/3       |        51% |        90%
 double_well |   3 |      1.8pp | 3/3       |        49% |        89%
    stoch_lv |   4 |      1.7pp | 3/4       |        51% |        89%
         fhn |   4 |      1.4pp | 4/4       |        50% |        90%
   sindy_sde |   5 |      1.2pp | 5/5       |        49% |        90%
--------------------------------------------------------------------
SBC-pass parameters: 31/32  |  mean calib-err: 1.3pp  (target: low err, cov50~50%, cov90~90%)

--- per-parameter SBC-p (p>0.05 = calibrated) ---
  linear_gaussian: m1:0.33  m2:0.17  m3:0.70  m4:0.12
            ou: theta:0.69  sigma:0.60
          seir: beta1:0.18  beta2:0.36  alpha:0.50  gamma_r:0.95  gamma_d:0.07
           gbm: mu:0.41  sigma:0.30
           cir: a:0.80  b:0.33  sigma:0.96
   double_well: theta1:0.68  theta2:0.58  sigma:0.56
      stoch_lv: alpha:0.14  beta:0.04  delta:0.30  gamma:0.67
           fhn: a:0.99  b:0.80  eps:0.38  I:0.45
     sindy_sde: c0:0.47  c1:0.19  c2:0.37  c3:0.11  sigma:0.56
```
