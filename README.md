# 

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

# Observing formation and early evolution of contrails formed by IAGOS aircraft using high-resolution LEO satellite imagery

This repository contains the code and data necessary to reproduce the figures from our paper:

[![Pre-print]()]()


<!--- Add here the hyperlink to the finalized version of the paper/journal-article related to this project  
    (the DOI link provided by the journal publisher after peer-review acceptance) (if applicable) e.g.:  

    This repository is the official implementation of the following paper.

    * Paper title: [Paper Title](https://doi.org/YYMM.NNNNN)
--> 


## Description

The study focuses on observing the formation and early evolution of contrails produced by IAGOS aircraft using high-resolution LEO satellite imagery, including Sentinel-2 and Landsat.

The full dataset, including satellite images, annotations, full IAGOS measurements, and ERA5 data, can be found at *(add link to zenodo data)*.  

In addition to the raw data, I made it possible to reproduce the final results of the paper without the need for the full dataset. However, some figures will not be rendered without the full dataset. See the local CSV file: [data/landsat_sentinel_collocations_20260212.csv](data/landsat_sentinel_collocations_20260212.csv).

## History


## Authors or Maintainers

<!--- Provide information about authors, maintainers and collaborators specifying contact details and role within the project, e.g.:   
    * Full name ([@GitHub username](https://github.com/username), [ORCID](https://doi.org/...), email address, institution/employer (role)  
-->

* Thymen Woldhuis ([@ThymenW](https://github.com/thymenW), t.woldhuis-1@tudelft.nl  )

## Code

- **Figure reproduction:**  
  Code to reproduce the figures from the paper is located in: [notebooks/paper_reproduction](notebooks/paper_reproduction), with the figures being saved in [notebooks/paper_reproduction/figures](notebooks/paper_reproduction/figures).



## Requirements  

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

<!-- 
## References

<!--- Provide links to applicable references    
--> 

 -->

## Citation
If you want to cite this repository in your research paper, please use the following information:  Reference: []()  


<!--- Make the repository citable 
    * If you will be using the Zenodo-Github integration, add the following reference and the DOI of the Zenodo repository:

        If you want to cite this repository in your research paper, please use the following information:
        Reference: [Making Your Code Citable](https://guides.github.com/activities/citable-code/)  

    * If you will be using the 4TU.ResearchData-Github integration, add the following reference and the DOI of the 4TU.ResearchData repository:

        If you want to cite this repository in your research paper, please use the following information:   
        Reference: [Connecting 4TU.ResearchData with Git](https://data.4tu.nl/info/about-your-data/getting-started)   
-->


<!-- 
## Would you like to contribute?

<!--- Add here how you would like others to contribute to this project (e.g. forking, opening issues only, etc.)

    * Do not forget to mention how others can specify how they contributed to the project (e.g., add their names in a separate list of Contributors in the README; add their contributions in separate files specifying their copyright attribution at the top of the source files as commented text; etc.)  
--> -->