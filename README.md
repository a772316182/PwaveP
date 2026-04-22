# PwaveP

> Official Code of 'PWAVEP: Purifying Imperceptible Adversarial Perturbations in 3D Point Clouds via Spectral Graph
> Wavelets'

## 📖 Overview

We created this repository to share some simple and efficient PyTorch implementations of graph wavelet transforms.
Noticing a lack of tools with built-in automatic differentiation, we organized the core components into a few utility
classes.

- **`WaveletTransCheb.py`**: Handles Chebyshev-approximation transforms and their inverses.
- **`WaveletTrans.py`**: Handles eigen-decomposition transforms and their inverses.

We sincerely hope these tools can help the community more easily adopt and benefit from this powerful analytical
technique.

Additionally, we've included a trainable, fully sparse graph wavelet filter. While this approach wasn't used in our
final paper, our experiments showed it delivered promising performance with modest memory usage. If you're curious,
please feel free to check it out in `src/trainable_wavelet_model/main.py`.

-----

## 💾 Installation

Before you begin, please ensure you have the following packages installed. You can typically install them using `pip`.

- `python==3.9.21`
- `pytorch==2.4.0`
- `pytorch-cuda==11.8`
- `torch-geometric==2.6.1`
- `pygsp2==2.0.6`
- `open3d==0.19.0`
- `pytorch3d==0.7.8`

-----

## 📂 Data Setup

The project relies on the following data structure:

- **Adversarial Examples**: Generated adversarial examples are stored in the `./attacked_data/` directory.

- **Pre-trained Models**: Our pre-trained model checkpoints are located in the `./ckpt/` directory.

- **Clean Data**: Due to its large size, the clean dataset is not included in the repository. Could you please download
  it from the link below and place the contents into a `./data/` directory

    - [**Download Clean Data from Google Drive
      **](https://drive.google.com/file/d/1BUP46fXOlLVeGKa7PT1CtBEi3s0HGHUi/view?usp=sharing)

-----

## ▶️ Usage

Here are the commands to run the evaluation scripts for the different models.

#### To evaluate PwaveP:

```shell
# For the new version
python protocols/defenders/run_pwavep.py
```

#### To evaluate PFourierP:

```shell
python protocols/defenders/run_pfourierp.py
```

If this repository is helpful for your research, please consider citing our paper:

```bibtex
@article{li2026pwavep,
  title={PWAVEP: Purifying Imperceptible Adversarial Perturbations in 3D Point Clouds via Spectral Graph Wavelets},
  author={Li, Haoran and Liu, Renyang and Liu, Hongjia and Wang, Chen and Yin, Long and Xu, Jian},
  journal={WWW},
  year={2026}
}
```
