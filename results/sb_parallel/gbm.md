# Honest scoreboard

```
honest scoreboard | 40000 sims, 12000 steps | 100 datasets x 400 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
           gbm |     24.84% |  10.04% |    10.48% |    11.23% | loses to ridge
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

gbm
           mu:  24.74% /  14.77% /  17.18% |  1.48x  
        sigma:  24.95% /   5.31% /   3.79% |  5.66x  

```
