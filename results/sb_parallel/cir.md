# Honest scoreboard

```
honest scoreboard | 40000 sims, 12000 steps | 100 datasets x 400 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
           cir |     24.14% |  11.06% |     8.27% |     8.50% | beats ridge
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

cir
            a:  22.29% /  20.37% /  15.42% |  1.44x  
            b:  24.18% /   6.86% /   4.62% |  5.33x  
        sigma:  25.95% /   5.95% /   4.77% |  4.89x  

```
