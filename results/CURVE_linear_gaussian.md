# learning curve: linear_gaussian

```
learning curve on 'linear_gaussian' | 40 datasets x 200 draws
fixed yardsticks -- prior-only 23.80%, ridge 17.33% (budget-independent)

 n_train  epochs |     err | contraction | per-parameter err
----------------------------------------------------------------------------------
    1500       6 |  21.26% |       1.04x | m1:24.9  m2:22.8  m3:17.5  m4:19.8
    3000      12 |  20.07% |       1.10x | m1:23.7  m2:21.9  m3:16.1  m4:18.6
    6000      24 |  11.85% |       1.86x | m1:16.4  m2:15.1  m3:7.4  m4:8.5
----------------------------------------------------------------------------------
error fell 44% from the smallest to the largest budget
the curve is still falling -- the budget was the binding constraint

per-parameter contraction across budgets (1.0 = still the prior):
           m1: 1.03x  1.04x  1.53x
           m2: 1.02x  1.03x  1.58x
           m3: 1.07x  1.15x  2.02x
           m4: 1.05x  1.17x  2.32x
```
