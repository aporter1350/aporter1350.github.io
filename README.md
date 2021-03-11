# CATS
Welcome to our blog for CATS (Constructing Algorithmic Trading Structures) — a market data project completed for STAT 359 in winter 2021. The goal of our project is to construct a data processing pipeline that can retreive, process, and analyze level 1 and level 2 market data.

## Pipeline Overview
The framework of our pipeline is described by the following:
<img width="726" alt="Screen Shot 2021-03-11 at 10 14 55 AM" src="https://user-images.githubusercontent.com/78179650/110818712-220b6280-8253-11eb-9587-64cbc6e78f34.png">

## Acquiring level 1 and level 2 data
People have been trying to predict the stock market for centuries. Many investment strategies involve using different types of market data to predict future stock behavior. With advancements in acquiring real time market book data, investors can now apply many different types of analyses to identify new trading opportunities that could be profitable for investors. But this comes at a cost, streaming data in real time requires a thoroughly vetted pipeline that can process and store data in a way that is flexible for use. This often requires customized hardware and a dedicated location for storing incoming streams of data.

The first part of our pipeline involved collecting level 1 and level 2 data. Data allocation was done using a cloud [platform](https://iexcloud.io/blog/how-to-get-market-data-in-python). IEX cloud is a platform that provides financial data to clients in order to use IEX cloud in python users must download [pyEx](https://github.com/timkpaine/pyEX) and create an account on IEX cloud in order to receive an API key for streaming. Level 1 data is fairly easy to acquire and can be done in python.

```
import pyEX as p
c = p.Client(api_token=YOUR API KEY)
sym='IEX' #stock symbol
timeframe='5y' #timeframe
df = c.chartDF(symbol=sym, timeframe=timeframe)
df.to_csv("~/.qlib/csv_data/5Y_IEX.csv")
```

To get the data in a format for qlib, open the terminal and type

```
python scripts/dump_bin.py dump_all --csv_path ~/.qlib/csv_data/IEX.csv --qlib_dir ~/.qlib/qlib_data/iex --include_fields close,high,low,open,volume --date_field_name date
```

Acquiring level 2 data often involves one of two methods: the first would be to pay a premium to vendors who calculate, organize, and collect ongoing data for you. The alternative method is to stream data for yourself, giving the user more control on incoming data streams. This involves the use of whats called a server sent event. This is a process that enables a user to receive automatic updates from the IEX website. What this means is that a user can create a link to a website using an API key. The script would then check on the website to see if a particular stock has changed. If it has the website will then send information back to the user usually in the form of a dictionary like object consisting of bid price, bid size, ask price, ask size, and the time. And this process can occur indefinitely but keep in mind that most users do not have the storage capacity to run this script forever, it also costs money to receive ongoing level 2 data. So unless you’re paying for a premium subscription, you’re fairly limited in how much data you can retrieve. Streaming can sometimes be considered more efficient because you are receiving the latest available data. Endpoints can be adjusted to specific intervals such as 1second, 5 seconds, or 1 minute.

```
tmp=[]
symbol='iex' #stock symbol
channels='book' #market book data
c.deepSSE(symbol,'book',on_data=tmp.append)
```

## Data Processing
One of the most important tools for data processing is Qlib, an AI-oriented investment platform. Extensive documentation for Qlib can be found [here](https://qlib.readthedocs.io/en/latest/), and the official repository can be found [here](https://github.com/microsoft/qlib). Qlib contains an extensive framework of modules that all serve different purposes in market data processing. However, these modules are designed to be standalone pieces of code, which in turn allows us to focus on the ones that are most relevant to our project.

### Dumping data into Qlib
Extensive information on how to input, or "dump" data into Qlib can be found in the above documentation and is also described above. However, in some cases  preprocessing of the data might be required in order to successfully dump data into Qlib. This might be the case when the data contains duplicate values, etc. An example of a dataset that presents this issue (sampleMSFTdata.csv), as well as a preprocessing script to fix this problem (qlibdump.py), can be found in the src folder.

### Qlib Processing Classes
Qlib provides many useful data-processing child classes. An exhaustive list can be found in the aforementioned documentation, but the ones that are most relevant to our project can be found in the processor (processor.py). The processor also contains some custom-built child classes that we created to process our specific datasets. For example, the FormatLevelTwo child class was created to reformat realtime level 2 data obtained from the IEX cloud service. An example of how these child classes can be used to process data is provided in the src file (processorexample.ipynb).

## Modeling

### Modeling Overview

### Our Models
After using Qlib to process our data into an acceptible format, we can use the features of XGBoost modeling to create models.


