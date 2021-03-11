import pandas as pd
data = pd.read_csv('MSFT_2004-10-31_2020-06-12_minute.csv')
data = data.drop_duplicates(subset='t', keep='first')
data.to_csv("MSFT.csv", index=False)
