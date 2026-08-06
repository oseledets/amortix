# Honest scoreboard

```
honest scoreboard | budget 2000/6 | 40 datasets x 200 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
linear_gaussian |     23.80% |  19.60% |    20.63% |     4.51% | loses to ridge
            ou |     24.17% |  10.89% |    19.31% |    12.91% | loses to ridge
------------------------------------------------------------------------------
amortized beats the ridge control in 0/2 cases

--- per parameter (prior-only / ridge / amortized) ---
a parameter whose amortized error sits at the prior-only level is not
being recovered at all, whatever the aggregate says

linear_gaussian
           m1:  25.77% /  26.38% /  24.90%  <-- no information
           m2:  22.86% /  18.76% /  22.76%  <-- no information
           m3:  22.23% /  18.21% /  17.13%
           m4:  24.33% /  15.04% /  17.73%

ou
        theta:  21.27% /  17.76% /  19.24%  <-- no information
           mu:  24.15% /   9.80% /  18.10%
        sigma:  27.09% /   5.11% /  20.61%
```
