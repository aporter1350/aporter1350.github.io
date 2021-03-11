import pyEX as p
c = p.Client(api_token=YOUR API KEY)
sym='IEX'
timeframe='1y'
#level 1 data
df = c.chartDF(symbol=sym, timeframe=timeframe)
df.to_csv("~/.qlib/csv_data/5Y_IEX.csv")
#level 2 data
tmp=[]
c.deepSSE('iex','book',on_data=tmp.append)
df=pd.DataFrame.from_dict(tmp)
df.to_csv(''~/.qlib/csv_data/L2_IEX.csv')
#python scripts/dump_bin.py dump_all --csv_path ~/.qlib/csv_data/IEX.csv --qlib_dir ~/.qlib/qlib_data/iex --include_fields close,high,low,open,volume --date_field_name date
#python scripts/dump_bin.py dump_all --csv_path ~/.qlib/csv_data/L2_IEX.csv --qlib_dir ~/.qlib/qlib_data/L2 --include_fields price,rsi,size,type --date_field_name time --freq 1s
