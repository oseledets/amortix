# Honest scoreboard

```
honest scoreboard | budget 3000/8 | 40 datasets x 300 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
     sindy_sde |     24.84% |  18.41% |    23.62% |    32.38% | loses to ridge
------------------------------------------------------------------------------
amortized beats the ridge control in 0/1 cases

--- per parameter: prior-only / ridge / amortized | contraction | verdict ---
contraction = prior sd / posterior sd (1.0 = the posterior IS the prior).
A flat posterior alone is ambiguous, so the ridge control arbitrates:
  NO INFO   - the posterior is flat AND the ridge cannot beat the prior
              => the data does not contain the parameter; scoring it is noise
  WE FAILED - the posterior is flat but the ridge DOES extract the parameter
              => the information is there and our model missed it

sindy_sde
           c0:  21.36% /  17.64% /  17.55% |  1.07x  NO INFO
           c1:  25.16% /  14.20% /  23.28% |  1.00x  WE FAILED
           c2:  22.98% /  25.97% /  23.72% |  0.99x  NO INFO
           c3:  29.72% /  26.76% /  30.70% |  0.98x  NO INFO
        sigma:  24.99% /   7.46% /  22.87% |  1.00x  WE FAILED

no information in the data (3): sindy_sde.c0, sindy_sde.c2, sindy_sde.c3
OUR FAILURES -- ridge extracts what we do not (2): sindy_sde.c1, sindy_sde.sigma
```
