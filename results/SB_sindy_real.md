# Honest scoreboard

```
honest scoreboard | budget 20000/50 | 80 datasets x 400 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
     sindy_sde |     24.19% |  17.50% |    17.77% |    29.62% | loses to ridge
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
           c0:  23.21% /  15.12% /  13.20% |  1.77x  
           c1:  22.67% /  20.42% /  16.17% |  1.39x  
           c2:  24.83% /  23.48% /  23.49% |  1.02x  PRIOR-LIMITED
           c3:  27.90% /  21.38% /  26.82% |  1.03x  WIDTH WRONG
        sigma:  22.32% /   7.09% /   9.15% |  2.68x  

prior-limited, not a defect (1): sindy_sde.c2  -- score these by SBC / distance to the true posterior, not by MAE
POSTERIOR TOO WIDE -- the ridge locates what we leave at the prior (1): sindy_sde.c3
```
