import pandas as pd
data = pd.read_csv('sampleMSFTdata.csv')
data = data.drop_duplicates(subset='t', keep='first')
data.to_csv("MSFT.csv", index=False)
