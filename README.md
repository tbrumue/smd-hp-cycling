# Large-scale monitoring of residential heat pump cycling using smart meter data

**Author:** Tobias Brudermüller (Brudermueller), Bits to Energy Lab, ETH Zurich: <tbrudermuell@ethz.ch>

This repository contains the Python code and data for the [following paper](https://www.sciencedirect.com/science/article/pii/S030626192301098X): 

> *Brudermueller, T., Kreft, M., Fleisch, E., & Staake, T. (2023). Large-scale monitoring of residential heat pump cycling using smart meter data. Applied Energy, 350, 121734.*

For detailed explanations about underlying assumptions, and implementations please refer to this source. 
Further, if you make use of this repository of paper, please **use the following citation**: 
```
@article{BRUDERMUELLER2023121734,
title = {Large-scale monitoring of residential heat pump cycling using smart meter data},
journal = {Applied Energy},
volume = {350},
pages = {121734},
year = {2023},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2023.121734},
url = {https://www.sciencedirect.com/science/article/pii/S030626192301098X},
author = {Tobias Brudermueller and Markus Kreft and Elgar Fleisch and Thorsten Staake},
keywords = {Smart meter data, Heat pump, Cycling behavior, outlier detection, Residential buildings, Energy efficiency, Appliance monitoring, Real-world operation, Machine learning}
}
```

---

### Abstract 

Heat pumps play an essential role in decarbonizing the building sector, but their electricity consumption can vary significantly across buildings. This variability is closely related to their cycling behavior (i.e., the frequency of on–off transitions), which is also an indicator for improper sizing and non-optimal settings and can affect a heat pump’s lifetime. Up to now it has been unclear which cycling behaviors are typical and atypical for heat pump operation in the field and importantly, there is a lack of methods to identify heat pumps that cycle atypically. Therefore, in this study we develop a method to monitor heat pumps with energy measurements delivered by common smart electricity meters, which also cover heat pumps without network connectivity. We show how smart meter data with 15-minute resolution can be used to extract key indicators about heat pump cycling and outline how atypical behavior can be detected after controlling for outdoor temperature. Our method is robust across different building characteristics and varying times of observation, does not require contextual information, and can be implemented with existing smart meter data, making it suitable for real-world applications. Analyzing 503 heat pumps in Swiss households over a period of 21 months, we further describe behavioral differences with respect to building and heat pump characteristics and study the relationship between heat pumps’ cycling behavior, energy efficiency, and appropriate sizing. Our results show that outliers in cycling behavior are more than twice as common for air-source heat pumps than for ground-source heat pumps.

---

### Installation 

If you want to use your Python interpreter directly, please check that you have the packages pip-installed which are listed in the file ```installation/requirements.yml```. Otherwise, if you want to create an anaconda environment named ```hp_cycling```, you can use the following commands: 

1. Navigating into installation folder: ```cd [....]/installation```
2. Installing environment: ```conda env create -n hp_cycling -f requirements.yml``` 
3. Opening environment for a session: ```conda activate hp_cycling```
4. Close environment after a session: ```conda deactivate```

**NOTE:**: The code also uses parallel processing and therefore uses the ```functools``` and ```multiprocess``` packages included in the standard Python library.

---

### Data

The whole original data set used in the paper, which includes the smart meter data of multiple heat pumps, cannot be shared. However, an example of a single heat pump with 15-minute resolution is provided in the file ```data/01_smart_meter_data/smd_hp.csv```. This data is used in the notebook ```notebooks/01_extract_cycling_behavior.ipynb```. Please further note that the the daily average temperatures of the nearest weather stations are also already added as a separate column. 

Next to this, you can find the daily metrics of all heat pumps extracted from the original smart meter data in the folder ```data/02_extracted_cycles```. The CSV-files are named according to the ID of each heat pump. Please note that the column ```cycle_energy_sum``` refers to the energy in kWh after removing the baseload. Hence, it does not necessarily refer to the total energy of the heat pump within a cycle. 

Additionally, also, the daily metrics of all heat pumps can be found in a separate folder named ```data/03_extracted_daily_metrics``` alongside the households' meta data (again using the same IDs). The files provided in ```data/03_extracted_daily_metrics``` are used in the notebook ```notebooks/02_evaluate_cycling_behavior.ipynb```. Please note that the column ```Energy_kWh``` covers the total energy in kWh that heat pump consumed at each date, i.e., it also includes the baseload and refers to the energy consumption before filtering.

---

### Usage 

Probably, the best way to understand this repository is to proceed in the following order: 

1. Read the [paper](https://www.sciencedirect.com/science/article/pii/S030626192301098X), which describes the methodology in detail.
2. Skim the data provided in the ```data``` folder. 
3. Check out the notebooks provided in the ```notebooks``` folder. 
4. Apply the methods to your own data by adjusting the sample code provided in the notebooks and by taking a closer look at the actual function implementations in ```src/helper.py```.