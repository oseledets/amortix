# Honest scoreboard

```
honest scoreboard | budget 12000/40 | 100 datasets x 400 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
linear_gaussian |     24.19% |  19.15% |     4.51% |     4.19% | beats ridge
            ou |     24.14% |  10.66% |     9.03% |    12.17% | beats ridge
           gbm |     24.84% |  10.85% |    11.37% |    14.18% | loses to ridge
------------------------------------------------------------------------------
amortized beats the ridge control in 2/3 cases

--- per parameter (prior-only / ridge / amortized) ---
a parameter whose amortized error sits at the prior-only level is not
being recovered at all, whatever the aggregate says

linear_gaussian
           m1:  26.15% /  19.89% /   4.60%
           m2:  24.09% /  23.27% /   6.30%
           m3:  23.64% /  18.39% /   3.66%
           m4:  22.87% /  15.07% /   3.49%

ou
        theta:  22.29% /  17.29% /  15.34%
           mu:  24.18% /   9.63% /   7.13%
        sigma:  25.95% /   5.07% /   4.63%

gbm
           mu:  24.74% /  16.65% /  17.95%
        sigma:  24.95% /   5.05% /   4.79%
```
