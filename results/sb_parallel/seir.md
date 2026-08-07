# Honest scoreboard

```
honest scoreboard | 40000 sims, 12000 steps | 100 datasets x 400 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
          seir |     24.83% |  19.97% |    14.24% |    24.04% | beats ridge
------------------------------------------------------------------------------
amortized beats the ridge control in 1/1 cases

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

seir
        beta1:  24.30% /  16.49% /  11.52% |  2.10x  
        beta2:  23.36% /  21.68% /  12.88% |  1.82x  
        alpha:  24.76% /  21.64% /  23.25% |  1.08x  PRIOR-LIMITED
      gamma_r:  28.16% /  17.78% /  14.68% |  1.61x  
      gamma_d:  23.56% /  22.26% /   8.83% |  2.52x  

prior-limited, not a defect (1): seir.alpha  -- score these by SBC / distance to the true posterior, not by MAE
```
