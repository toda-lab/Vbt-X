# VBT-X
VBT-X is a diversity-aware fairness testing approach for black-box machine learning models.
VBT-X improves verification-based testing (VBT), another fairness testing approach, with a sampling technique.
We first reimplement VBT, referring to [the source code of VBT](https://github.com/arnabsharma91/fairCheck), and then implement the sampling technique.

Refer to our paper [SSBSE'22](https://link.springer.com/chapter/10.1007/978-3-031-21251-2_3) for more technical details.
The experimental results of the evaluation of VBT-X are available in
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10252552.svg)](https://doi.org/10.5281/zenodo.10252552).

## Contributors
VBT-X was proposed by [Zhenjiang Zhao](https://zhenjiang-zhao.github.io/), Takahisa Toda and Takashi Kitamura.
The implementation was performed by [Zhenjiang Zhao](https://zhenjiang-zhao.github.io/).

## Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [How to cite](#how-to-cite)
- [License](#license)

## Requirements
The experiments were based on `Python 3.8.10`.
We recommend using a virtual environment to run and test VBT-X,
to avoid dependency conflicts between packages.
We show two ways to create a **virtual environment**.
- If you already have `Python 3.8.10` installed, you can use `venv` to create a virtual environment.
- If not, you can
  - install `Python 3.8.10` use [`pyenv`](https://github.com/pyenv/pyenv), which enables you to install and manage multiple different versions of Python,
and then create a virtual environment using `venv`,
  - or use [`conda`](https://github.com/conda/conda) to directly create a virtual environment that already contains `Python 3.8.10`.

### Using venv
Make sure `Python 3.8.10` and `venv` are installed.

Create a virtual environment called `env_vbtx`:
```
python3 -m venv env_vbtx
```

Activate the environment:
```
source env_vbtx/bin/activate
```

Install required packages:
```
pip install -r requirements.txt   
```

### Using conda
Make sure `conda` is installed.

Create a virtual environment called `env_vbtx`, and install `Python 3.8.10` and `pip` in it:
```
conda create --name env_vbtx python=3.8.10 pip
```

Activate the environment:
```
conda activate env_vbtx
```

Install required packages:
```
pip install -r requirements.txt   
```

## Usage
Use the following command to run VBT-X:
```
python exp.py <dataset> <protected_attr> <model> <vbtx_version> [runtime] [loop_times]
```
The possible values for each parameter are listed below:
- dataset: _Adult_, _Credit_, _Bank_
- protected_attr: _sex_, _race_, _age_
- model: _LogReg_, _NB_, _RanForest_, _DecTree_
- vbtx_version: _naive_, _improveds10_, _improved_
- runtime: the running time in seconds (_default=1200_)
- loop_times: the number of repeated runs (_default=31_)

The parameter 'vbtx_version' specifies the version of VBT-X to be used:
_naive_ refers to `Basic VBT-X(s=10)`,
_improveds10_ refers to `VBT-X(s=10)`
and _improved_ refers to `VBT-X`.

The folder [`FairnessTestCases`](FairnessTestCases) contains 12 machine learning models prepared for fairness testing.
The folder [`Datasets`](Datasets) contains 3 training datasets, on which the 12 models were trained.  

### Example
As an example, the following command means to use `VBT-X` to perform a fairness testing on the configuration _(Adult_, _sex_, _NB)_ within 60 seconds:
```
python exp.py Adult sex NB improved 60 1
```

### Outputs
Each run outputs two `.csv` files, located in two folders `DiscData` and `TestData`, respectively.
The results contained in each folder are as follows:
- [`DiscData`](DiscData): the results of detected discriminatory instances
- [`TestData`](TestData): the results of generated test cases

For the running example, suppose we obtain a result file `DiscData/improved-NB-Adult-sex-60-0.csv`, and the first six rows of this file are shown below:
```
$ head -6 DiscData/improved-NB-Adult-sex-60-0.csv
1,35,54,1,6,1,0,2,1,50,1,9,74,1
1,35,54,1,6,1,0,2,0,50,1,9,74,0
3,0,28,1,4,55,0,0,1,74,7,81,58,1
3,0,28,1,4,55,0,0,0,74,7,81,58,0
5,26,11,3,4,0,4,0,1,3,3,7,57,1
5,26,11,3,4,0,4,0,0,3,3,7,57,0
```

For interpreting this file, 
a row represents an individual,
and a pair of two consecutive rows represents a discriminatory instance.
For example, the pair of row 1 and row 2 is a discriminatory instance, the pair of row 3 and row 4 is another discriminatory instance, and so on.

An individual (i.e., a row) is represented as a sequence of attribute values.
The order of these attributes is the same as the order of the attributes of the training data.
For example, you can find the list of attributes for _Adult_ with the following command:
```
$ head -1 Datasets/Adult.csv
age,workclass,fnlwgt,education,martial_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,Class
```

The results in `DiscData` are interpreted in the same as above,
except that a pair of two consecutive rows represents a test case.

## How to cite
If you use the VBT-X, please cite our paper [SSBSE'22](https://link.springer.com/chapter/10.1007/978-3-031-21251-2_3). 

Bibtex:
```
@InProceedings{zhao2022ssbse,
author="Zhao, Zhenjiang and Toda, Takahisa and Kitamura, Takashi",
title="Efficient Fairness Testing Through Hash-Based Sampling",
booktitle="Search-Based Software Engineering",
year="2022",
pages="35--50"
}
```
## License
VBT-X is licensed under [The MIT License](LICENSE).