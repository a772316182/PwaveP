# PwaveP

### packages
plz install the following packages:
- torch-geometric           2.6.1
- pygsp2                    2.0.6
- open3d                    0.19.0
- pytorch3d                 0.7.8
- pytorch                   2.4.0
- pytorch-cuda              11.8
- python                    3.9.21

### data
> the generated adversarial examples are stored in:
> 
> ./attacked_data/

> the pre-fitted clean models are stored in:
> 
> ./ckpt/

> the clean data, due to its size, is not provided in github, plz download it from:
> 
> https://drive.google.com/file/d/1BUP46fXOlLVeGKa7PT1CtBEi3s0HGHUi/view?usp=sharing
> 
> and put it in the ./data/ directory.



### usage
to eval PwaveP, run:
```shell
python protocols/defenders/run_new_pwavep.py
# or
python protocols/defenders/run_old_pwavep.py
```

to eval PFourierP, run:
```shell
python protocols/defenders/run_pfourierp.py
```

to eval PointCVAR, run:
```shell
python protocols/defenders/run_pointcvar.py
```