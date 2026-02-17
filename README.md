# Observing formation and early evolution of contrails formed by IAGOS aircraft using high-resolution LEO satellite imagery

![Banner](notebooks/mozaic_low_res.png)

This repository is the official implementation of the following paper: *(add reference to paper once published)*

## Description

The study focuses on observing the formation and early evolution of contrails produced by IAGOS aircraft using high-resolution LEO satellite imagery, including Sentinel-2 and Landsat.

The full dataset, including satellite images, annotations, full IAGOS measurements, and ERA5 data, can be found at [4TU Research Portal](https://doi.org/10.4121/2d66d65e-8041-4435-ab3c-0af3fdfc5d23). 

In addition to the raw data, I made it possible to reproduce most of the final results of the paper without the need for the full dataset. However, some figures or flights will not be rendered. See the local CSV file: [landsat_sentinel_collocations_20260216.csv](data/landsat_sentinel_collocations_20260216.csv).


## Authors or Maintainers

* Thymen Woldhuis ([@ThymenW](https://github.com/ThymenW), ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0000-2237-1839](https://orcid.org/0009-0000-2237-1839), t.woldhuis-1@tudelft.nl, Delft University of Technology.

## Usage

### Reproducing the figures from the paper
The code to reproduce the figures in the paper are located in [notebooks/paper_reproduction](notebooks/paper_reproduction), with the figures being saved in [notebooks/paper_reproduction/figures](notebooks/paper_reproduction/figures). Most figures can be recreated without the full dataset. For the others, first download the full dataset at [4TU Research Portal](https://doi.org/10.4121/2d66d65e-8041-4435-ab3c-0af3fdfc5d23). 

Note: for employees of TU Delft, I can invite you to the U:drive with the dataset. Please send me a message in that case. Change `DATASET_LOCATION` in all notebooks to:

```python
DATASET_LOCATION = "/Volumes/staff-umbrella/IagosSentinelColocations/data"
```

### Further analysis
To experiment with the data there are some notebooks given in [notebooks](notebooks/). 
*note: not added yet*

## Requirements and Installation

To install requirements go to the directory and run

  ```bash
  pip install -e .
  ```

or 

  ```bash
  poetry install
  ```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

Copyright notice:  

TU Delft hereby disclaims all copyright interest in the program “[name_program]” (provide one line description of the content or function) written by the Author(s).  

© 2026, Thymen Woldhuis


## Citation
If you want to cite this repository in your research paper, please use the following information: *add reference to 4TU once published*
