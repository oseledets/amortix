# ablation: cir / base

```
ablation on 'cir' over base=['data', 'full']
budget 512/1, dim=64, SBC 20x40, seeds=[0]

[base=data]  calib-err=23.9±0.0pp  mean SBC-pass=0.0/3
           a: mean SBC-p=0.018  passed 0/1 seeds
           b: mean SBC-p=0.031  passed 0/1 seeds
       sigma: mean SBC-p=0.000  passed 0/1 seeds

[base=full]  calib-err=15.6±0.0pp  mean SBC-pass=3.0/3
           a: mean SBC-p=0.083  passed 1/1 seeds
           b: mean SBC-p=0.130  passed 1/1 seeds
       sigma: mean SBC-p=0.196  passed 1/1 seeds

=> lowest calib-err: base=full (15.6pp)
```
