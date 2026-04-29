# HFCSM-Text
Arbitrary shaped text detection via high-frequency fusion and circular sequence modeling

## Model Architecture

![Model Architecture](assets/model_architecture.svg)

## Prerequisites

To run this repo successfully, it is recommended with:

- **Linux** (Ubuntu 22.04)
- **Python 3.10**
- **NVIDIA GPU with CUDA support**
- **PyTorch**
- **OpenCV**
- **mamba-ssm**
- > **Note:** The `requirements.txt` file lists the required packages for the original development environment. HFCSM-Text may only require a subset of these packages to run properly. For more details, please refer to `requirements.txt`.

## Code Release Note

This repository currently provides the main implementation of HFCSM-Text.  
For some auxiliary configuration files, dataset processing scripts, and evaluation details, please refer to the excellent work of [TextBPN++](https://github.com/GXYM/TextBPN-Plus-Plus), on which this project is partly based.

## Model Performance

| Dataset | P (%) | R (%) | F-measure (%) | FPS |
|---|---:|---:|---:|---:|
| Total-Text | 93.11 | 86.36 | 89.61 | 15.29 |
| CTW-1500 | 89.88 | 83.08 | 86.35 | 17.12 |
| MSRA-TD500 | 93.07 | 86.13 | 89.48 | 19.89 |



## Acknowledgements

We sincerely thank the authors of [TextBPN++](https://github.com/GXYM/TextBPN-Plus-Plus) and [HS-FPN](https://github.com/ShiZican/HS-FPN) for their excellent work and open-source contributions. Their codebases provide important foundations and inspirations for this project.
