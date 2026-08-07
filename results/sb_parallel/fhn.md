# Honest scoreboard

```
honest scoreboard | 40000 sims, 12000 steps | 100 datasets x 400 draws
          case | prior-only |   ridge | amortized | classical | verdict
------------------------------------------------------------------------------
           fhn |     24.19% |  15.50% |    10.10% |    11.57% | beats ridge
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

fhn
            a:  26.15% /  15.06% /  13.15% |  1.89x  
            b:  24.09% /  20.97% /  14.15% |  1.65x  
          eps:  23.64% /   8.82% /   2.33% |  8.81x  
            I:  22.87% /  17.16% /  10.78% |  2.07x  

```
