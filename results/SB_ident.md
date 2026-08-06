# Honest scoreboard

```
honest scoreboard | budget 3000/8 | 40 datasets x 300 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
     sindy_sde |     24.84% |  18.41% |    22.66% |    32.38% | loses to ridge
------------------------------------------------------------------------------
amortized beats the ridge control in 0/1 cases

--- per parameter: prior-only / ridge / amortized | contraction | verdict ---
contraction = prior sd / posterior sd (1.0 = the posterior IS the prior).
A wide posterior is NOT a failure: where the likelihood is flat the correct
posterior is the prior. What a flat posterior does mean is that error-to-truth
is capped at the prior's 25% and says nothing about method quality there.
The ridge control disambiguates the two flat cases:
  PRIOR-LIMITED - flat, and the ridge cannot do better either
                  => near-prior is the right answer; judge it by SBC, not MAE
  WIDTH WRONG   - flat, but the ridge locates the parameter
                  => the true posterior is narrow and ours is not: a real error

sindy_sde
           c0:  21.36% /  17.64% /  17.91% |  1.04x  PRIOR-LIMITED
           c1:  25.16% /  14.20% /  22.51% |  1.02x  WIDTH WRONG
           c2:  22.98% /  25.97% /  22.53% |  1.01x  PRIOR-LIMITED
           c3:  29.72% /  26.76% /  29.51% |  0.99x  PRIOR-LIMITED
        sigma:  24.99% /   7.46% /  20.84% |  1.03x  WIDTH WRONG

prior-limited, not a defect (3): sindy_sde.c0, sindy_sde.c2, sindy_sde.c3  -- score these by SBC / distance to the true posterior, not by MAE
POSTERIOR TOO WIDE -- the ridge locates what we leave at the prior (2): sindy_sde.c1, sindy_sde.sigma
```
