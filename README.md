## Overview
<div style="text-align: center;">
  <img src="overview_CellDialog.png" alt="Alt Text">
</div>

## Data
Data is available at [uniprot](https://www.uniprot.org/), [GEO](https://www.ncbi.nlm.nih.gov/geo/).

## Package Environment
Install python3 for running this code. And these packages should be satisfied:
* tensorflow == 2.5.0
* keras == 2.7.0
* pandas == 1.1.4
* numpy == 1.19.5
* scikit-learn == 1.0.1
* matplotlib == 0.1.5
* xgboost == 1.6.2
* lightgbm == 3.3.0
* KTBoost == 0.2.2
* gpboost == 0.7.10


## Feature Acquisition
[iFeature](https://github.com/Superzchen/iFeature)

## CCC analysis tools
[cellphonedb](https://github.com/Teichlab/cellphonedb),
[NATMI](https://github.com/asrhou/NATMI)

## Usage
First, run the model, the default 5 fold cross-validation, get LRI pairs. Or the user can user-specified LRI database directly, skip this step to the third step.
```
python code/CellDialog.py

```
The second step, The data format for processing cancer types is /example/xxx.csv .

The third step, Run three calculation methods[(cell expression)(expression product)(expression thresholding)]. (Note: the user-specified database only needs to replace the LRI.csv file and the corresponding format in the file.)
```
python example/The cell expression calculation method code.py
python example/The expression product calculation method code.py
python example/The expression thresholding calculation method code.py
```
The final step, the three point estimation method is used to combine the results of three methods, the three point estimation method are detailed in the paper.

## Hardware Environment
This code was run on a computer with the following specifications:
* CPU: AMD EPYC 7302
* Memory: 256GB DDR4
* GPU: GeForce RTX 2080 Ti

However, the minimal requirements for running this protocol are:
* CPU: Intel Core i7-11800H
* Memory: 16GB DDR4
* GPU: NVIDIA GeForce RTX 3050Ti
* Storage: At least 10GB available


