import processor
import pandas as pd
#Input level 2 data
df = pd.read_csv("rawlevel2data.csv")
#Run raw data through child class
my_processor = processor.FormatLevelTwo()
df = my_processor(df)
#Output processed data
df.to_csv("processedlevel2data.csv")
