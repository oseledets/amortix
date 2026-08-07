# Honest scoreboard

```
honest scoreboard | 2000 sims, 200 steps | 20 datasets x 100 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
           gbm |     26.34% |  11.80% |    21.63% |    10.91% | loses to ridge
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
           mu:  30.11% /  15.53% /  24.99% |  1.20x  
        sigma:  22.56% /   8.07% /  18.26% |  1.15x  

```
