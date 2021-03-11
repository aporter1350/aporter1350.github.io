# CATS
Welcome to our blog for CATS (Constructing Algorithmic Trading Structures) — a market data project completed for STAT 359 in winter 2021. The goal of our project is to construct a data processing pipeline that can retreive, process, and analyze level 1 and level 2 market data.

## Pipeline Overview
The framework of our pipeline is described by the following:
<img width="726" alt="Screen Shot 2021-03-11 at 10 14 55 AM" src="https://user-images.githubusercontent.com/78179650/110818712-220b6280-8253-11eb-9587-64cbc6e78f34.png">

## Data Processing
One of the most important tools for data processing is Qlib, an AI-oriented investment platform. Extensive documentation for Qlib can be found [here](https://qlib.readthedocs.io/en/latest/), and the official repository can be found [here](https://github.com/microsoft/qlib).

### Dumping data into Qlib
Extensive information on how to input, or "dump" data into Qlib can be found in the above documentation. However, in some cases further preprocessing of the data might be required in order to successfully dump data into Qlib. This might be the case when the data contains duplicate values, etc. An example of a dataset that presents this problem (sampleMSFTdata.csv), as well as a preprocessing script to fix this problem (qlibdump.py), can be found in the src folder.

### Qlib Processing Classes
Qlib provides many useful data-processing child classes. An exhaustive list can be found in the aforementioned documentation, but the ones that are most relevant to our project can be found in the processor (processor.py). The processor also contains some custom-built child classes that we created to process our specific datasets. For example, the FormatLevelTwo child class was created to reformat realtime level 2 data obtained from the IEX cloud service. 

## Modeling


### Modeling Overview

### Our Models


