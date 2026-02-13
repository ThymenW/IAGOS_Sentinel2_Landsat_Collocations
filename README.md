# Observing formation and early evolution of contrails formed by IAGOS aircraft using high-resolution LEO satellite imagery

![Banner](notebooks/mozaic_low_res.png)

## Overview

This repository contains the code and data necessary to reproduce the figures from our paper:

**DOI:** *(to be added)*

The study focuses on observing the formation and early evolution of contrails produced by IAGOS aircraft using high-resolution LEO satellite imagery, including Sentinel-2 and Landsat.

## Data

The full dataset, including satellite images, annotations, full IAGOS measurements, and ERA5 data, can be found at *(add link to zenodo data)*.  

In addition to the raw data, I made it possible to reproduce the final results of the paper without the need for the full dataset. However, some figures will not be rendered without the full dataset. See the local CSV file: [data/landsat_sentinel_collocations_20260212.csv](data/landsat_sentinel_collocations_20260212.csv).

## Code

- **Figure reproduction:**  
  Code to reproduce the figures from the paper is located in: [notebooks/paper_reproduction](notebooks/paper_reproduction), with the figures being saved in [notebooks/paper_reproduction/figures](notebooks/paper_reproduction/figures)

## Adding New Satellite Collocations

To add new collocations between aircraft and satellite data, you can:  

1. Use the `pycontrails` function for collocating flights with Sentinel-2 and Landsat imagery.  
2. Contact *(Thymen Woldhuis)* for access to the source code for additional processing if needed.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.
